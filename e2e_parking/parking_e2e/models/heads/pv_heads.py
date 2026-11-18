"""PV（透视视图）三个头：重建（度量深度）、图像分割、图像检测。7 路针孔和 4 路鱼眼都在原图上做。"""
from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...registry import HEADS, LOSSES
from ..layers import conv_bn_relu
from ..losses import centernet_focal_loss
from .base import BaseHead


def min_pool_depth(depth: torch.Tensor, stride: int) -> torch.Tensor:
    """(B, N, H, W) 稀疏深度 -> (B, N, H/s, W/s)，每块取最近的有效深度（0 表示无效）。"""
    B, N, H, W = depth.shape
    h, w = H // stride, W // stride
    d = depth[:, :, :h * stride, :w * stride].reshape(B, N, h, stride, w, stride)
    d = torch.where(d > 0, d, torch.full_like(d, float("inf")))
    d = d.amin(dim=(3, 5))
    return torch.where(torch.isfinite(d), d, torch.zeros_like(d))


@HEADS.register_module()
class PVDepthHead(BaseHead):
    """重建 = 度量深度，直接监督 LSS 的深度分布（BEVDepth 的接法），本身没有参数。

    真值 gt_depth_{group} (B, N, H, W)，单位米，0 为无效。针孔是 z-depth，鱼眼是 range，
    与相机模型的 depth_type 一致。自车车身、天空、强反光应在数据侧置 0。
    """

    def __init__(self, groups: Sequence[str] = ("pinhole", "fisheye"), ce_weight: float = 1.0,
                 l1_weight: float = 0.5, group_weights: Optional[Dict[str, float]] = None):
        super().__init__()
        self.groups = list(groups)
        self.ce_weight = ce_weight
        self.l1_weight = l1_weight
        self.group_weights = group_weights or {}

    def loss(self, preds, feats, batch):
        out = {}
        for g in self.groups:
            key = f"gt_depth_{g}"
            if key not in batch or g not in feats["depth_logits"]:
                continue
            logits = feats["depth_logits"][g].float()                 # (B, N, D, h, w)
            meta = feats["depth_meta"][g]
            B, N, D, h, w = logits.shape
            gt = min_pool_depth(batch[key].float(), meta["stride"])[:, :, :h, :w]
            bins = torch.floor((gt - meta["dmin"]) / meta["step"]).long()
            valid = (gt > 0) & (bins >= 0) & (bins < D)
            gw = self.group_weights.get(g, 1.0)
            if not valid.any():
                out[f"loss_depth_ce_{g}"] = logits.sum() * 0
                continue
            target = torch.where(valid, bins, torch.full_like(bins, -100))
            ce = F.cross_entropy(logits.flatten(0, 1), target.flatten(0, 1), ignore_index=-100)
            centers = meta["bins"].view(1, 1, D, 1, 1).to(logits)
            exp_depth = (logits.softmax(2) * centers).sum(2)
            l1 = (exp_depth - gt).abs()[valid].mean()
            out[f"loss_depth_ce_{g}"] = gw * self.ce_weight * ce
            out[f"loss_depth_l1_{g}"] = gw * self.l1_weight * l1
            with torch.no_grad():
                out[f"absrel_{g}"] = ((exp_depth - gt).abs() / gt.clamp(min=1e-3))[valid].mean()
        return out

    def predict(self, preds, feats, batch):
        res = {}
        for g in self.groups:
            if g not in feats["depth_logits"]:
                continue
            logits = feats["depth_logits"][g]
            D = logits.shape[2]
            centers = feats["depth_meta"][g]["bins"].view(1, 1, D, 1, 1).to(logits)
            res[f"depth_{g}"] = (logits.softmax(2) * centers).sum(2)
        return res


@HEADS.register_module()
class PVSegHead(BaseHead):
    """原图语义分割（stride 8 特征），类别与 BEV 地面布局对齐。鱼眼有效圆外的像素标 255。"""

    def __init__(self, in_channels: int = 256, num_classes: int = 8, mid_channels: int = 128,
                 groups: Sequence[str] = ("pinhole", "fisheye"),
                 loss=dict(type="CrossEntropyDiceLoss", num_classes=8)):
        super().__init__()
        self.groups = list(groups)
        self.num_classes = num_classes
        self.net = nn.Sequential(conv_bn_relu(in_channels, mid_channels), conv_bn_relu(mid_channels, mid_channels),
                                 nn.Conv2d(mid_channels, num_classes, 1))
        self.loss_fn = LOSSES.build(loss)

    def forward(self, feats):
        out = {}
        for g in self.groups:
            if g in feats["pv"]:
                x = feats["pv"][g]
                B, N = x.shape[:2]
                out[g] = self.net(x.flatten(0, 1)).unflatten(0, (B, N))
        return out

    def loss(self, preds, feats, batch):
        out = {}
        for g, logits in preds.items():
            key = f"gt_pvseg_{g}"
            if key not in batch:
                continue
            gt = batch[key].long()
            up = F.interpolate(logits.flatten(0, 1), size=gt.shape[-2:], mode="bilinear", align_corners=False)
            out[f"loss_pvseg_{g}"] = self.loss_fn(up, gt.flatten(0, 1))
        return out

    def predict(self, preds, feats, batch):
        return {f"pvseg_{g}": l.argmax(2) for g, l in preds.items()}


@HEADS.register_module()
class PVDetHead(BaseHead):
    """CenterNet 式图像检测：热图 + 偏移 + 尺寸 + 方向。

    主要对象是 BEV 框不擅长的点状结构：车位角点（关键点，带进入方向）、地锁、车门。
    gt_pvdet_{group}: list[B] of list[N] of (M, 7) = [cls, cx, cy, w, h, dir_cos, dir_sin]（输入图像像素）。
    关键点 w = h = 0；没有方向的目标 dir 写 (0, 0)。
    """

    def __init__(self, in_channels: int = 256, num_classes: int = 4, mid_channels: int = 128, stride: int = 8,
                 groups: Sequence[str] = ("pinhole", "fisheye"), keypoint_sigma: float = 1.0,
                 box_sigma_ratio: float = 0.15, min_sigma: float = 0.8,
                 offset_weight: float = 1.0, size_weight: float = 0.1, dir_weight: float = 0.5,
                 max_det: int = 64, score_thr: float = 0.3):
        super().__init__()
        self.groups = list(groups)
        self.num_classes = num_classes
        self.stride = stride
        self.kp_sigma, self.box_sigma_ratio, self.min_sigma = keypoint_sigma, box_sigma_ratio, min_sigma
        self.w_off, self.w_size, self.w_dir = offset_weight, size_weight, dir_weight
        self.max_det, self.score_thr = max_det, score_thr

        def branch(cout):
            return nn.Sequential(conv_bn_relu(in_channels, mid_channels), nn.Conv2d(mid_channels, cout, 1))

        self.heat = branch(num_classes)
        self.offset = branch(2)
        self.size = branch(2)
        self.dir = branch(2)
        nn.init.constant_(self.heat[-1].bias, -2.19)

    def forward(self, feats):
        out = {}
        for g in self.groups:
            if g not in feats["pv"]:
                continue
            x = feats["pv"][g]
            B, N = x.shape[:2]
            xf = x.flatten(0, 1)
            out[g] = {k: getattr(self, k)(xf).unflatten(0, (B, N)) for k in ("heat", "offset", "size", "dir")}
        return out

    def _targets(self, gt, h, w, device):
        heat = torch.zeros(self.num_classes, h, w, device=device)
        if gt is None or gt.numel() == 0:
            return heat, None
        gt = gt.to(device).float()
        cls = gt[:, 0].long()
        c = gt[:, 1:3] / self.stride
        wh = gt[:, 3:5] / self.stride
        ci = c.floor().long()
        keep = (ci[:, 0] >= 0) & (ci[:, 0] < w) & (ci[:, 1] >= 0) & (ci[:, 1] < h) & (cls >= 0) & (cls < self.num_classes)
        if not keep.any():
            return heat, None
        cls, c, wh, ci, dirv = cls[keep], c[keep], wh[keep], ci[keep], gt[keep, 5:7]
        is_kp = (wh <= 0).any(1)
        sigma = torch.where(is_kp, torch.full_like(wh[:, 0], self.kp_sigma),
                            (wh[:, 0] * wh[:, 1]).clamp(min=1e-6).sqrt() * self.box_sigma_ratio).clamp(min=self.min_sigma)
        xs = torch.arange(w, device=device, dtype=torch.float32)
        ys = torch.arange(h, device=device, dtype=torch.float32)
        d2 = (xs[None, None, :] - ci[:, 0, None, None].float()) ** 2 + (ys[None, :, None] - ci[:, 1, None, None].float()) ** 2
        g = torch.exp(-d2 / (2 * sigma[:, None, None] ** 2))
        M = g.shape[0]
        heat = heat.view(self.num_classes, -1).scatter_reduce(
            0, cls[:, None].expand(M, h * w), g.view(M, -1), reduce="amax", include_self=True).view(self.num_classes, h, w)
        reg = dict(ci=ci, off=c - ci.float(), logwh=torch.log(wh.clamp(min=1e-3)), is_kp=is_kp,
                   dir=dirv, has_dir=dirv.norm(dim=1) > 0.5)
        return heat, reg

    def loss(self, preds, feats, batch):
        out = {}
        for g, p in preds.items():
            key = f"gt_pvdet_{g}"
            if key not in batch:
                continue
            B, N, K, h, w = p["heat"].shape
            heats, offs, sizes, dirs = [], [], [], []
            for b in range(B):
                for n in range(N):
                    gt = batch[key][b][n] if len(batch[key][b]) > n else None
                    heat, reg = self._targets(gt, h, w, p["heat"].device)
                    heats.append(heat)
                    if reg is None:
                        continue
                    iy, ix = reg["ci"][:, 1], reg["ci"][:, 0]
                    po = p["offset"][b, n][:, iy, ix].t()
                    ps = p["size"][b, n][:, iy, ix].t()
                    pd = p["dir"][b, n][:, iy, ix].t()
                    offs.append((po - reg["off"]).abs().sum(1))
                    box = ~reg["is_kp"]
                    if box.any():
                        sizes.append((ps[box] - reg["logwh"][box]).abs().sum(1))
                    if reg["has_dir"].any():
                        pdn = F.normalize(pd[reg["has_dir"]].float(), dim=1)
                        dirs.append((pdn - reg["dir"][reg["has_dir"]]).abs().sum(1))
            heat_t = torch.stack(heats).view(B, N, K, h, w)
            out[f"loss_pvdet_heat_{g}"] = centernet_focal_loss(p["heat"].float(), heat_t)
            zero = p["offset"].sum() * 0
            out[f"loss_pvdet_off_{g}"] = self.w_off * torch.cat(offs).mean() if offs else zero
            out[f"loss_pvdet_size_{g}"] = self.w_size * torch.cat(sizes).mean() if sizes else zero
            out[f"loss_pvdet_dir_{g}"] = self.w_dir * torch.cat(dirs).mean() if dirs else zero
        return out

    @torch.no_grad()
    def predict(self, preds, feats, batch):
        res = {}
        for g, p in preds.items():
            B, N, K, h, w = p["heat"].shape
            heat = p["heat"].sigmoid().flatten(0, 1)
            peak = F.max_pool2d(heat, 3, 1, 1) == heat
            scores, idx = (heat * peak).flatten(1).topk(min(self.max_det, K * h * w), dim=1)
            cls = idx // (h * w)
            pix = idx % (h * w)
            ys, xs = pix // w, pix % w
            off = p["offset"].flatten(0, 1).flatten(2).gather(2, pix[:, None].expand(-1, 2, -1))
            size = p["size"].flatten(0, 1).flatten(2).gather(2, pix[:, None].expand(-1, 2, -1)).exp()
            dirv = F.normalize(p["dir"].flatten(0, 1).flatten(2).gather(2, pix[:, None].expand(-1, 2, -1)), dim=1)
            cx = (xs + off[:, 0]) * self.stride
            cy = (ys + off[:, 1]) * self.stride
            dets = torch.stack([cls.float(), scores, cx, cy, size[:, 0] * self.stride, size[:, 1] * self.stride,
                                dirv[:, 0], dirv[:, 1]], -1)                # (BN, M, 8)
            res[f"pvdet_{g}"] = [d[d[:, 1] > self.score_thr] for d in dets]   # 长度 B*N
        return res
