"""GOD（通用障碍物）三个头：occ、occ flow、SDF。

三个头都是隐式解码器：在粗体素特征（默认 0.2 m）上三线性采样，拼上子体素位置编码，MLP 输出。
这样监督可以落在任意分辨率的查询点上（5 m 内 5 cm、5 m 外 10 cm），SDF 的零等值面也能落在格子内部，
规划碰撞项可以沿轨迹在任意点查询 SDF。

查询点由数据侧从多分辨率标签网格里采样，未知体素不采样（不参与损失）：
  occ_pts (B, N, 3)、occ_label (B, N)  0 空 / 1 占据 / 255 忽略、occ_sem (B, N) 255 忽略
  flow_pts (B, N, 3)、flow_val (B, N, 3) m/s、flow_mask (B, N)
  sdf_pts (B, N, 3)、sdf_val (B, N) 米（体素内为负）、sdf_mask (B, N)
"""
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call

from ...geometry import Grid3D, sample_voxel
from ...registry import HEADS
from ..layers import MLP, fourier_encode
from .base import BaseHead


class ImplicitVoxelHead(BaseHead):
    def __init__(self, grid, in_channels: int = 32, hidden: int = 128, num_layers: int = 3,
                 out_dim: int = 1, pe_freqs: int = 3, pred_chunk: int = 200000):
        super().__init__()
        self.grid = Grid3D.from_cfg(grid)
        self.pe_freqs = pe_freqs
        self.pred_chunk = pred_chunk
        self.mlp = MLP(in_channels + 3 + 3 * 2 * pe_freqs, hidden, out_dim, num_layers)

    def _inputs(self, voxel, pts):
        feat = sample_voxel(voxel, pts, self.grid)
        lo = pts.new_tensor(self.grid.lower)
        res = pts.new_tensor(self.grid.res)
        # 频率取 2π·2^k / res，周期是体素尺寸的整数分之一，跨格连续
        pe = fourier_encode(2.0 * (pts - lo) / res, self.pe_freqs)
        return torch.cat([feat, self.grid.normalize(pts).to(feat.dtype), pe.to(feat.dtype)], -1)

    def decode(self, voxel, pts, frozen: bool = False):
        """frozen=True 时梯度只流向 pts（规划碰撞项用），不更新本头参数和体素特征。"""
        if frozen:
            x = self._inputs(voxel.detach(), pts)
            params = {k: v.detach() for k, v in self.mlp.named_parameters()}
            return functional_call(self.mlp, params, (x,))
        return self.mlp(self._inputs(voxel, pts))

    @torch.no_grad()
    def decode_dense(self, voxel, pred_grid: Optional[Grid3D] = None):
        """在 pred_grid（默认与特征网格相同）的体素中心查询，返回 (B, out_dim, Z, Y, X)。"""
        g = pred_grid or self.grid
        centers = g.voxel_centers(voxel.device).view(1, -1, 3).expand(voxel.shape[0], -1, -1)
        outs = [self.decode(voxel, centers[:, i:i + self.pred_chunk]) for i in range(0, centers.shape[1], self.pred_chunk)]
        out = torch.cat(outs, 1)
        return out.transpose(1, 2).reshape(voxel.shape[0], -1, g.nz, g.ny, g.nx)


@HEADS.register_module()
class OccHead(ImplicitVoxelHead):
    """空 / 占据二分类 + 可选的占据细分类（车辆、立柱、路沿、其他）。细分类失败不影响通用占据。"""

    def __init__(self, grid, in_channels: int = 32, hidden: int = 128, num_layers: int = 3,
                 num_sem: int = 4, pos_weight: float = 2.0, sem_weight: float = 0.5,
                 pred_grid: Optional[dict] = None, ignore_index: int = 255):
        super().__init__(grid, in_channels, hidden, num_layers, 1 + num_sem)
        self.num_sem = num_sem
        self.pos_weight = pos_weight
        self.sem_weight = sem_weight
        self.ignore_index = ignore_index
        self.pred_grid = Grid3D.from_cfg(pred_grid) if pred_grid else None

    def loss(self, preds, feats, batch):
        pts, label = batch["occ_pts"].float(), batch["occ_label"].long()
        out = self.decode(feats["voxel"], pts).float()
        logit = out[..., 0]
        known = label != self.ignore_index
        res = {}
        if not known.any():
            res["loss_occ"] = out.sum() * 0
            return res
        pw = logit.new_tensor(self.pos_weight)
        res["loss_occ"] = F.binary_cross_entropy_with_logits(logit[known], label[known].float(), pos_weight=pw)
        if self.num_sem > 0 and "occ_sem" in batch:
            sem = batch["occ_sem"].long()
            m = (label == 1) & (sem != self.ignore_index)
            res["loss_occ_sem"] = self.sem_weight * F.cross_entropy(out[..., 1:][m], sem[m]) if m.any() else out.sum() * 0
        with torch.no_grad():
            occ, free = label == 1, label == 0
            pred = logit > 0
            res["occ_recall"] = (pred & occ).sum() / occ.sum().clamp(min=1)
            res["occ_free_fp"] = (pred & free).sum() / free.sum().clamp(min=1)
        return res

    @torch.no_grad()
    def predict(self, preds, feats, batch):
        out = self.decode_dense(feats["voxel"], self.pred_grid)
        res = dict(occ_prob=out[:, 0].sigmoid())
        if self.num_sem > 0:
            res["occ_sem"] = out[:, 1:].argmax(1)
        return res


@HEADS.register_module()
class OccFlowHead(ImplicitVoxelHead):
    """占据体素的 3D 速度（m/s）。动态物体的流来自检测跟踪的框速度，静态为 0。"""

    def __init__(self, grid, in_channels: int = 32, hidden: int = 128, num_layers: int = 3,
                 dynamic_weight: float = 4.0, dynamic_thr: float = 0.2, pred_grid: Optional[dict] = None):
        super().__init__(grid, in_channels, hidden, num_layers, 3)
        self.dynamic_weight = dynamic_weight
        self.dynamic_thr = dynamic_thr
        self.pred_grid = Grid3D.from_cfg(pred_grid) if pred_grid else None

    def loss(self, preds, feats, batch):
        pts, gt, m = batch["flow_pts"].float(), batch["flow_val"].float(), batch["flow_mask"].bool()
        flow = self.decode(feats["voxel"], pts).float()
        if not m.any():
            return dict(loss_occ_flow=flow.sum() * 0)
        err = F.smooth_l1_loss(flow[m], gt[m], reduction="none", beta=0.1).sum(-1)
        w = 1.0 + self.dynamic_weight * (gt[m].norm(dim=-1) > self.dynamic_thr).float()
        res = dict(loss_occ_flow=(err * w).sum() / w.sum())
        with torch.no_grad():
            res["flow_epe"] = (flow[m] - gt[m]).norm(dim=-1).mean()
        return res

    @torch.no_grad()
    def predict(self, preds, feats, batch):
        return dict(occ_flow=self.decode_dense(feats["voxel"], self.pred_grid))


@HEADS.register_module()
class SDFHead(ImplicitVoxelHead):
    """截断符号距离：体素内部为负、空侧为正，输出 τ·tanh(raw/τ)。

    - 截断带 |d| < τ 内算 L1；带外目标夹到 ±τ，以 outside_weight 小权重监督，只为稳住远处的符号。
    - 符号一致率作为日志项（表面两侧探针的预测符号与标签符号相同的比例）。符号大面积翻转时打开 sign_weight。
    - Eikonal 默认关闭：稀疏表面标签下它会和空洞打架。场在带内不单调时再小权重打开。
    该头默认留在部署图里，供规划的在线拒绝器查询间隙。
    """

    def __init__(self, grid, in_channels: int = 32, hidden: int = 128, num_layers: int = 3,
                 tau: float = 0.4, outside_weight: float = 0.1, sign_weight: float = 0.0,
                 eikonal_weight: float = 0.0, pred_grid: Optional[dict] = None):
        super().__init__(grid, in_channels, hidden, num_layers, 1)
        self.tau = tau
        self.outside_weight = outside_weight
        self.sign_weight = sign_weight
        self.eikonal_weight = eikonal_weight
        self.pred_grid = Grid3D.from_cfg(pred_grid) if pred_grid else None

    def _activate(self, raw):
        return self.tau * torch.tanh(raw / self.tau)

    def query(self, voxel, pts, frozen: bool = False):
        return self._activate(self.decode(voxel, pts, frozen)[..., 0].float())

    def loss(self, preds, feats, batch):
        pts, d, m = batch["sdf_pts"].float(), batch["sdf_val"].float(), batch["sdf_mask"].bool()
        use_eik = self.eikonal_weight > 0 and self.training
        if use_eik:
            pts = pts.clone().requires_grad_(True)
        raw = self.decode(feats["voxel"], pts)[..., 0].float()
        pred = self._activate(raw)
        res = {}
        if not m.any():
            res["loss_sdf"] = pred.sum() * 0
            return res
        band = m & (d.abs() < self.tau)
        outside = m & ~band
        target = d.clamp(-self.tau, self.tau)
        err = (pred - target).abs()
        loss = err[band].mean() if band.any() else pred.sum() * 0
        if outside.any() and self.outside_weight > 0:
            loss = loss + self.outside_weight * err[outside].mean()
        res["loss_sdf"] = loss
        if self.sign_weight > 0 and band.any():
            res["loss_sdf_sign"] = self.sign_weight * F.binary_cross_entropy_with_logits(
                raw[band] / self.tau, (d[band] > 0).float())
        if use_eik:
            sel = band & (d.abs() < 0.5 * self.tau)
            grad = torch.autograd.grad(pred[sel].sum(), pts, create_graph=True)[0] if sel.any() else None
            res["loss_sdf_eikonal"] = (self.eikonal_weight * ((grad[sel].norm(dim=-1) - 1) ** 2).mean()
                                       if grad is not None else pred.sum() * 0)
        with torch.no_grad():
            probe = band & (d.abs() > 0.02)
            res["sdf_sign_acc"] = ((pred > 0) == (d > 0))[probe].float().mean() if probe.any() else pred.new_tensor(1.0)
            res["sdf_band_l1"] = err[band].mean() if band.any() else pred.new_tensor(0.0)
        return res

    @torch.no_grad()
    def predict(self, preds, feats, batch):
        if self.pred_grid is None:
            return {}
        return dict(sdf=self._activate(self.decode_dense(feats["voxel"], self.pred_grid)[:, 0]))
