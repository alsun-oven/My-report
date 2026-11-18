"""自车坐标系：x 前、y 左、z 上，原点在后轴中心的地面投影。

BEV 张量布局 (B, C, NY, NX)，体素张量布局 (B, C, NZ, NY, NX)。
格子 [iz, iy, ix] 的中心为 (xmin + (ix + 0.5) * res_x, ...)。
"""
from typing import Sequence, Tuple, Union

import torch


class Grid3D:
    def __init__(self, xbound: Sequence[float], ybound: Sequence[float], zbound: Sequence[float]):
        self.xbound = tuple(float(v) for v in xbound)
        self.ybound = tuple(float(v) for v in ybound)
        self.zbound = tuple(float(v) for v in zbound)

    @classmethod
    def from_cfg(cls, cfg: Union["Grid3D", dict]) -> "Grid3D":
        return cfg if isinstance(cfg, Grid3D) else cls(**cfg)

    def to_dict(self) -> dict:
        return dict(xbound=self.xbound, ybound=self.ybound, zbound=self.zbound)

    @staticmethod
    def _n(b) -> int:
        return int(round((b[1] - b[0]) / b[2]))

    @property
    def nx(self) -> int:
        return self._n(self.xbound)

    @property
    def ny(self) -> int:
        return self._n(self.ybound)

    @property
    def nz(self) -> int:
        return self._n(self.zbound)

    @property
    def shape_zyx(self) -> Tuple[int, int, int]:
        return self.nz, self.ny, self.nx

    @property
    def lower(self) -> Tuple[float, float, float]:
        return self.xbound[0], self.ybound[0], self.zbound[0]

    @property
    def upper(self) -> Tuple[float, float, float]:
        return self.xbound[1], self.ybound[1], self.zbound[1]

    @property
    def res(self) -> Tuple[float, float, float]:
        return self.xbound[2], self.ybound[2], self.zbound[2]

    def to_index(self, pts: torch.Tensor):
        """pts (..., 3) -> ix, iy, iz (long) 与是否在范围内的掩码。"""
        lo = pts.new_tensor(self.lower)
        res = pts.new_tensor(self.res)
        idx = torch.floor((pts - lo) / res).long()
        ix, iy, iz = idx.unbind(-1)
        valid = (ix >= 0) & (ix < self.nx) & (iy >= 0) & (iy < self.ny) & (iz >= 0) & (iz < self.nz)
        return ix, iy, iz, valid

    def to_index_2d(self, xy: torch.Tensor):
        lo = xy.new_tensor(self.lower[:2])
        res = xy.new_tensor(self.res[:2])
        idx = torch.floor((xy - lo) / res).long()
        ix, iy = idx.unbind(-1)
        valid = (ix >= 0) & (ix < self.nx) & (iy >= 0) & (iy < self.ny)
        return ix, iy, valid

    def normalize(self, pts: torch.Tensor) -> torch.Tensor:
        """米制坐标 -> grid_sample 的 [-1, 1]（align_corners=False）。最后一维 (x, y[, z])。"""
        d = pts.shape[-1]
        lo = pts.new_tensor(self.lower[:d])
        hi = pts.new_tensor(self.upper[:d])
        return (pts - lo) / (hi - lo) * 2.0 - 1.0

    def denormalize(self, norm: torch.Tensor) -> torch.Tensor:
        d = norm.shape[-1]
        lo = norm.new_tensor(self.lower[:d])
        hi = norm.new_tensor(self.upper[:d])
        return (norm + 1.0) * 0.5 * (hi - lo) + lo

    def bev_centers(self, device=None, dtype=torch.float32) -> torch.Tensor:
        """(NY, NX, 2) 每个 BEV 格子中心的 (x, y)。"""
        xs = torch.arange(self.nx, device=device, dtype=dtype) * self.xbound[2] + self.xbound[0] + 0.5 * self.xbound[2]
        ys = torch.arange(self.ny, device=device, dtype=dtype) * self.ybound[2] + self.ybound[0] + 0.5 * self.ybound[2]
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        return torch.stack([xx, yy], dim=-1)

    def voxel_centers(self, device=None, dtype=torch.float32) -> torch.Tensor:
        """(NZ, NY, NX, 3)。"""
        xy = self.bev_centers(device, dtype)
        zs = torch.arange(self.nz, device=device, dtype=dtype) * self.zbound[2] + self.zbound[0] + 0.5 * self.zbound[2]
        xy = xy[None].expand(self.nz, -1, -1, -1)
        z = zs[:, None, None, None].expand(-1, self.ny, self.nx, 1)
        return torch.cat([xy, z], dim=-1)
