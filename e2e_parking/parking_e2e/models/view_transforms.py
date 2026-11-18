"""多相机组 Lift-Splat：7 路针孔 + 4 路鱼眼抬升到同一个近场体素网格。

每个相机组有自己的相机模型、深度桶和 DepthNet。深度分布既用于抬升，也被 PVDepthHead（重建）监督，
所以它属于视角变换本身，部署时随 BEV 保留。
"""
from typing import Dict, Sequence

import torch
import torch.nn as nn

from ..geometry import Grid3D, transform_points
from ..registry import CAMERA_MODELS, VIEW_TRANSFORMS
from .layers import ResBlock, conv_bn_relu


class DepthNet(nn.Module):
    def __init__(self, cin, mid, context_channels, depth_bins, calib_dim):
        super().__init__()
        self.reduce = conv_bn_relu(cin, mid)
        # 相机参数调制特征（BEVDepth 的 camera-aware 做法的简化版）
        self.calib_mlp = nn.Sequential(nn.Linear(calib_dim, mid), nn.ReLU(inplace=True), nn.Linear(mid, mid))
        self.depth_branch = nn.Sequential(ResBlock(mid, mid), ResBlock(mid, mid), nn.Conv2d(mid, depth_bins, 1))
        self.context_branch = nn.Sequential(ResBlock(mid, mid), nn.Conv2d(mid, context_channels, 1))

    def forward(self, x, calib_vec):
        x = self.reduce(x)
        x = x * torch.sigmoid(self.calib_mlp(calib_vec))[:, :, None, None]
        return self.depth_branch(x), self.context_branch(x)


@VIEW_TRANSFORMS.register_module()
class MultiGroupLSS(nn.Module):
    def __init__(self, grid, cam_groups: Dict[str, dict], in_channels: int = 256, mid_channels: int = 256,
                 voxel_channels: int = 32, bev_channels: int = 128):
        """cam_groups: {name: dict(camera_model=dict(type=...), depth_bins=(min, max, step), stride=16)}"""
        super().__init__()
        self.grid = Grid3D.from_cfg(grid)
        self.groups = list(cam_groups)
        self.voxel_channels = voxel_channels
        self.cam_models = {}
        self.strides = {}
        self.depth_cfg = {}
        self.depthnets = nn.ModuleDict()
        for name, gcfg in cam_groups.items():
            cam = CAMERA_MODELS.build(gcfg["camera_model"])
            self.cam_models[name] = cam
            self.strides[name] = int(gcfg.get("stride", 16))
            dmin, dmax, step = gcfg["depth_bins"]
            n = int(round((dmax - dmin) / step))
            self.depth_cfg[name] = (float(dmin), float(step), n)
            self.register_buffer(f"bins_{name}", dmin + step * (torch.arange(n, dtype=torch.float32) + 0.5),
                                 persistent=False)
            calib_dim = 4 + cam.num_distortion + 12
            self.depthnets[name] = DepthNet(in_channels, mid_channels, voxel_channels, n, calib_dim)
        nz = self.grid.nz
        self.bev_proj = nn.Sequential(
            nn.Conv2d(voxel_channels * nz, bev_channels, 1, bias=False),
            nn.BatchNorm2d(bev_channels), nn.ReLU(inplace=True))

    def depth_bins(self, group: str) -> torch.Tensor:
        return getattr(self, f"bins_{group}")

    def _calib_vec(self, group, calib, img_hw):
        H, W = img_hw
        intr = calib["intrinsics"].float()
        parts = [intr / intr.new_tensor([W, H, W, H])]
        k = self.cam_models[group].num_distortion
        if k > 0:
            dist = calib.get("distortion")
            dist = intr.new_zeros(intr.shape[:-1] + (k,)) if dist is None else dist.float()[..., :k]
            parts.append(dist)
        parts.append(calib["cam2ego"].float()[..., :3, :].flatten(-2))
        return torch.cat(parts, -1)

    def frustum_points(self, group, calib, B, N, h, w):
        """-> pts (B, N, D, h, w, 3) 自车系，valid (B, N, h, w)。"""
        stride = self.strides[group]
        dev = calib["intrinsics"].device
        u = (torch.arange(w, device=dev, dtype=torch.float32) + 0.5) * stride - 0.5
        v = (torch.arange(h, device=dev, dtype=torch.float32) + 0.5) * stride - 0.5
        vv, uu = torch.meshgrid(v, u, indexing="ij")
        uv = torch.stack([uu, vv], -1).expand(B, N, h, w, 2)
        intr = calib["intrinsics"].float()[:, :, None, None, :]
        dist = calib.get("distortion")
        if dist is not None and dist.shape[-1] > 0:
            dist = dist.float()[:, :, None, None, :].expand(B, N, h, w, -1)
        else:
            dist = None
        rays, valid = self.cam_models[group].unproject(uv, intr.expand(B, N, h, w, 4), dist)
        d = self.depth_bins(group).view(1, 1, -1, 1, 1, 1)
        pts_cam = rays[:, :, None] * d                                     # (B, N, D, h, w, 3)
        T = calib["cam2ego"].float()[:, :, None, None, None]
        return transform_points(T, pts_cam), valid

    def voxel_pool(self, pts, feats, valid, out_flat, B):
        ix, iy, iz, inb = self.grid.to_index(pts)
        keep = inb & valid[:, :, None]
        nz, ny, nx = self.grid.shape_zyx
        bidx = torch.arange(B, device=pts.device).view(B, 1, 1, 1, 1).expand_as(ix)
        flat = ((bidx * nz + iz) * ny + iy) * nx + ix
        out_flat.index_add_(0, flat[keep], feats[keep].to(out_flat.dtype))

    def forward(self, feats: Dict[str, torch.Tensor], calibs: Dict[str, dict], img_shapes: Dict[str, Sequence[int]]):
        """feats[g] (B, N, C, h, w)。返回 voxel (B, Cv, Z, Y, X)、bev (B, Cb, Y, X)、depth_logits[g] (B, N, D, h, w)。"""
        B = next(iter(feats.values())).shape[0]
        nz, ny, nx = self.grid.shape_zyx
        C = self.voxel_channels
        dev = next(iter(feats.values())).device
        out_flat = torch.zeros(B * nz * ny * nx, C, device=dev, dtype=torch.float32)
        depth_logits = {}
        for g in self.groups:
            if g not in feats:
                continue
            x = feats[g]
            _, N, Cin, h, w = x.shape
            calib = calibs[g]
            calib_vec = self._calib_vec(g, calib, img_shapes[g]).flatten(0, 1).to(x.dtype)
            dlogit, ctx = self.depthnets[g](x.flatten(0, 1), calib_vec)
            prob = dlogit.softmax(1)
            D = prob.shape[1]
            vol = prob.unsqueeze(1) * ctx.unsqueeze(2)                     # (BN, C, D, h, w)
            vol = vol.view(B, N, C, D, h, w).permute(0, 1, 3, 4, 5, 2)    # (B, N, D, h, w, C)
            pts, valid = self.frustum_points(g, calib, B, N, h, w)
            self.voxel_pool(pts, vol, valid, out_flat, B)
            depth_logits[g] = dlogit.view(B, N, D, h, w)
        voxel = out_flat.view(B, nz, ny, nx, C).permute(0, 4, 1, 2, 3).contiguous()
        bev = self.bev_proj(voxel.flatten(1, 2))
        return dict(voxel=voxel, bev=bev, depth_logits=depth_logits)
