"""BEV 两个头：3D 分割（语义 + 高度带，2.5D）和 3D 检测（查询 + 匈牙利匹配）。"""
import math
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...geometry import Grid3D
from ...registry import HEADS, LOSSES, MATCHERS
from ..layers import MLP, DeformableDecoderLayer, conv_bn_relu, inverse_sigmoid
from .base import BaseHead


@HEADS.register_module()
class BEVSegHead(BaseHead):
    """每格预测语义类，以及占据的高度下沿、上沿（相对局部地面，米）。

    BEV 特征 0.2 m，upsample=4 输出到 5 cm。
    gt_bev_seg (B, Hs, Ws) long，255 忽略；gt_bev_height (B, 2, Hs, Ws)，无效处写 NaN。
    """

    def __init__(self, in_channels: int = 128, num_classes: int = 8, upsample: int = 4, mid_channels: int = 64,
                 empty_index: int = 0, height_weight: float = 1.0,
                 loss=dict(type="CrossEntropyDiceLoss", num_classes=8)):
        super().__init__()
        assert upsample >= 1 and (upsample & (upsample - 1)) == 0, "upsample must be power of 2"
        self.num_classes = num_classes
        self.empty_index = empty_index
        self.height_weight = height_weight
        self.stem = conv_bn_relu(in_channels, mid_channels)
        self.ups = nn.ModuleList([conv_bn_relu(mid_channels, mid_channels) for _ in range(int(math.log2(upsample)))])
        self.sem = nn.Conv2d(mid_channels, num_classes, 1)
        self.height = nn.Conv2d(mid_channels, 2, 1)
        self.loss_fn = LOSSES.build(loss)

    def forward(self, feats):
        x = self.stem(feats["bev"])
        for up in self.ups:
            x = up(F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False))
        return dict(sem=self.sem(x), height=self.height(x))

    def loss(self, preds, feats, batch):
        gt = batch["gt_bev_seg"].long()
        assert gt.shape[-2:] == preds["sem"].shape[-2:], \
            f"gt_bev_seg {tuple(gt.shape[-2:])} vs pred {tuple(preds['sem'].shape[-2:])}"
        out = dict(loss_bevseg=self.loss_fn(preds["sem"], gt))
        if "gt_bev_height" in batch and self.height_weight > 0:
            gh = batch["gt_bev_height"].float()
            m = (gt != self.empty_index) & (gt != self.loss_fn.ignore_index) & torch.isfinite(gh).all(1)
            if m.any():
                diff = (preds["height"].float() - torch.nan_to_num(gh)).abs()
                out["loss_bevseg_height"] = self.height_weight * diff.permute(0, 2, 3, 1)[m].mean()
            else:
                out["loss_bevseg_height"] = preds["height"].sum() * 0
        return out

    def predict(self, preds, feats, batch):
        return dict(bev_seg=preds["sem"].argmax(1), bev_height=preds["height"])


class BEVQueryDecoder(nn.Module):
    """可变形注意力的查询解码器，逐层细化参考点（中心）。BEV 检测和车位查询共用。"""

    def __init__(self, in_channels, embed_dims=256, num_queries=200, num_layers=3, num_heads=8,
                 num_points=4, ffn_dims=1024):
        super().__init__()
        self.d = embed_dims
        self.input_proj = nn.Conv2d(in_channels, embed_dims, 1)
        self.query = nn.Embedding(num_queries, 2 * embed_dims)
        self.ref_point = nn.Linear(embed_dims, 2)
        self.layers = nn.ModuleList([DeformableDecoderLayer(embed_dims, num_heads, num_points, ffn_dims)
                                     for _ in range(num_layers)])
        self.center_branches = nn.ModuleList([MLP(embed_dims, embed_dims, 2, 2) for _ in range(num_layers)])
        for br in self.center_branches:
            nn.init.zeros_(br.net[-1].weight)
            nn.init.zeros_(br.net[-1].bias)

    def forward(self, bev):
        x = self.input_proj(bev)
        B = x.shape[0]
        content, pos = self.query.weight.split(self.d, -1)
        q = content[None].expand(B, -1, -1)
        pos = pos[None].expand(B, -1, -1)
        ref = self.ref_point(pos).sigmoid()
        hs, centers = [], []
        for layer, br in zip(self.layers, self.center_branches):
            q = layer(q, pos, ref, x)
            new_ref = (br(q) + inverse_sigmoid(ref)).sigmoid()
            hs.append(q)
            centers.append(new_ref)
            ref = new_ref.detach()
        return hs, centers


@HEADS.register_module()
class BEVDetHead(BaseHead):
    """3D 检测。每个查询：类别（sigmoid focal）、中心、z、log 尺寸、sin/cos 朝向、速度、属性（地锁开/关）。

    速度只在 dynamic_classes 上监督。gt_boxes_3d: list[B] of (M, 9) = x, y, z, l, w, h, yaw, vx, vy；
    gt_labels_3d: list[B] of (M,)；gt_attr_3d: list[B] of (M,)，-1 表示该类没有属性。
    """

    def __init__(self, grid, in_channels: int = 128, embed_dims: int = 256, num_queries: int = 200,
                 num_classes: int = 7, num_attrs: int = 2, num_layers: int = 3, num_heads: int = 8,
                 num_points: int = 4, dynamic_classes: Sequence[int] = (0, 1, 2),
                 matcher=dict(type="HungarianMatcher3D"), loss_cls=dict(type="SigmoidFocalLoss"),
                 reg_weights: Sequence[float] = (1, 1, 1, 1, 1, 1, 1, 1, 0.5, 0.5), reg_weight: float = 0.25,
                 attr_weight: float = 0.5, max_det: int = 100, score_thr: float = 0.2):
        super().__init__()
        self.grid = Grid3D.from_cfg(grid)
        self.num_classes = num_classes
        self.num_attrs = num_attrs
        self.decoder = BEVQueryDecoder(in_channels, embed_dims, num_queries, num_layers, num_heads, num_points)
        L = num_layers
        self.cls_branches = nn.ModuleList([nn.Linear(embed_dims, num_classes) for _ in range(L)])
        self.reg_branches = nn.ModuleList([MLP(embed_dims, embed_dims, 8, 2) for _ in range(L)])
        self.attr_branches = nn.ModuleList([nn.Linear(embed_dims, num_attrs) for _ in range(L)]) if num_attrs else None
        for br in self.cls_branches:
            nn.init.constant_(br.bias, -4.6)
        self.register_buffer("dynamic_classes", torch.tensor(list(dynamic_classes), dtype=torch.long), persistent=False)
        self.register_buffer("reg_w", torch.tensor(reg_weights, dtype=torch.float32), persistent=False)
        self.matcher = MATCHERS.build(matcher)
        self.loss_cls = LOSSES.build(loss_cls)
        self.reg_weight = reg_weight
        self.attr_weight = attr_weight
        self.max_det, self.score_thr = max_det, score_thr

    def forward(self, feats):
        hs, centers = self.decoder(feats["bev"])
        out = dict(cls=[], center=[], reg=[], attr=[])
        for l, (q, c) in enumerate(zip(hs, centers)):
            out["cls"].append(self.cls_branches[l](q))
            out["center"].append(c)
            out["reg"].append(self.reg_branches[l](q))
            out["attr"].append(self.attr_branches[l](q) if self.attr_branches is not None else None)
        return out

    def _center_m(self, c):
        lo = c.new_tensor(self.grid.lower[:2])
        hi = c.new_tensor(self.grid.upper[:2])
        return c * (hi - lo) + lo

    def _center_norm(self, xy):
        lo = xy.new_tensor(self.grid.lower[:2])
        hi = xy.new_tensor(self.grid.upper[:2])
        return (xy - lo) / (hi - lo)

    def _decode(self, center, reg):
        """-> (…, 10) = x, y, z, log l, log w, log h, sin, cos, vx, vy"""
        return torch.cat([self._center_m(center), reg], -1)

    @staticmethod
    def _encode_gt(boxes):
        return torch.cat([boxes[:, :3], boxes[:, 3:6].clamp(min=1e-2).log(), boxes[:, 6:7].sin(),
                          boxes[:, 6:7].cos(), boxes[:, 7:9]], -1)

    def _filter_gt(self, boxes, labels, attrs):
        x, y = boxes[:, 0], boxes[:, 1]
        (x0, y0, _), (x1, y1, _) = self.grid.lower, self.grid.upper
        keep = (x >= x0) & (x < x1) & (y >= y0) & (y < y1) & (labels >= 0) & (labels < self.num_classes)
        return boxes[keep], labels[keep], attrs[keep] if attrs is not None else None

    def loss(self, preds, feats, batch):
        out = {}
        B = preds["cls"][0].shape[0]
        dev = preds["cls"][0].device
        gts = []
        for b in range(B):
            boxes = batch["gt_boxes_3d"][b].to(dev).float()
            labels = batch["gt_labels_3d"][b].to(dev).long()
            attrs = batch["gt_attr_3d"][b].to(dev).long() if "gt_attr_3d" in batch else None
            gts.append(self._filter_gt(boxes, labels, attrs))
        num_gt = max(sum(g[1].numel() for g in gts), 1)

        for l in range(len(preds["cls"])):
            cls_l, ctr_l, reg_l, attr_l = preds["cls"][l], preds["center"][l], preds["reg"][l], preds["attr"][l]
            cls_t = torch.zeros_like(cls_l)
            reg_losses, attr_losses = [], []
            for b, (boxes, labels, attrs) in enumerate(gts):
                qi, gi = self.matcher(cls_l[b], ctr_l[b], labels, self._center_norm(boxes[:, :2]))
                if qi.numel() == 0:
                    continue
                qi, gi = qi.to(dev), gi.to(dev)
                cls_t[b, qi, labels[gi]] = 1.0
                pred = self._decode(ctr_l[b, qi], reg_l[b, qi])
                tgt = self._encode_gt(boxes[gi])
                w = self.reg_w.expand_as(tgt).clone()
                is_dyn = (labels[gi][:, None] == self.dynamic_classes[None]).any(1)
                w[~is_dyn, 8:] = 0
                reg_losses.append(((pred - tgt).abs() * w).sum(1))
                if attr_l is not None and attrs is not None:
                    a = attrs[gi]
                    m = a >= 0
                    if m.any():
                        attr_losses.append(F.cross_entropy(attr_l[b, qi][m].float(), a[m], reduction="none"))
            out[f"loss_det_cls_d{l}"] = self.loss_cls(cls_l, cls_t, avg_factor=num_gt)
            zero = reg_l.sum() * 0
            out[f"loss_det_reg_d{l}"] = self.reg_weight * torch.cat(reg_losses).sum() / num_gt if reg_losses else zero
            if self.attr_branches is not None:
                out[f"loss_det_attr_d{l}"] = self.attr_weight * torch.cat(attr_losses).mean() if attr_losses else zero
        return out

    @torch.no_grad()
    def predict(self, preds, feats, batch):
        cls, ctr, reg, attr = preds["cls"][-1], preds["center"][-1], preds["reg"][-1], preds["attr"][-1]
        B, Q, K = cls.shape
        scores, idx = cls.sigmoid().flatten(1).topk(min(self.max_det, Q * K), dim=1)
        labels = idx % K
        qidx = idx // K
        results = []
        for b in range(B):
            box = self._decode(ctr[b, qidx[b]], reg[b, qidx[b]])
            yaw = torch.atan2(box[:, 6], box[:, 7])
            boxes = torch.cat([box[:, :3], box[:, 3:6].exp(), yaw[:, None], box[:, 8:10]], -1)
            keep = scores[b] > self.score_thr
            r = dict(boxes_3d=boxes[keep], scores_3d=scores[b][keep], labels_3d=labels[b][keep])
            if attr is not None:
                r["attr_3d"] = attr[b, qidx[b]].argmax(-1)[keep]
            results.append(r)
        return dict(det3d=results)
