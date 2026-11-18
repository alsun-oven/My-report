import torch
import torch.nn.functional as F

from .grid import Grid3D


def sample_voxel(voxel: torch.Tensor, pts: torch.Tensor, grid: Grid3D) -> torch.Tensor:
    """三线性采样体素特征。voxel (B, C, Z, Y, X)，pts (B, N, 3) 米制 -> (B, N, C)。范围外为 0。"""
    norm = grid.normalize(pts).to(voxel.dtype)
    out = F.grid_sample(voxel, norm[:, :, None, None, :], mode="bilinear",
                        padding_mode="zeros", align_corners=False)
    return out[:, :, :, 0, 0].transpose(1, 2)


def sample_bev(bev: torch.Tensor, xy: torch.Tensor, grid: Grid3D) -> torch.Tensor:
    """bev (B, C, Y, X)，xy (B, N, 2) -> (B, N, C)。"""
    norm = grid.normalize(xy).to(bev.dtype)
    out = F.grid_sample(bev, norm[:, :, None, :], mode="bilinear",
                        padding_mode="zeros", align_corners=False)
    return out[:, :, :, 0].transpose(1, 2)


def warp_bev(feat: torch.Tensor, T_src_from_dst: torch.Tensor, grid: Grid3D) -> torch.Tensor:
    """把 src 帧的 BEV 特征重采样到 dst 帧网格上（平面运动假设）。

    feat (B, C, Y, X) 或 (B, C, Z, Y, X)，T_src_from_dst (B, 4, 4)。
    体素输入把 Z 并进通道，按同一套 xy 位移重采样，z 不变。
    """
    is_voxel = feat.dim() == 5
    if is_voxel:
        B, C, Z, Y, X = feat.shape
        feat2d = feat.reshape(B, C * Z, Y, X)
    else:
        feat2d = feat
    B = feat2d.shape[0]
    xy = grid.bev_centers(feat.device, torch.float32)                      # (Y, X, 2)
    pts = torch.cat([xy, torch.zeros_like(xy[..., :1]), torch.ones_like(xy[..., :1])], -1)
    pts = pts.view(1, -1, 4).expand(B, -1, -1)                             # (B, YX, 4)
    src = (T_src_from_dst.float() @ pts.transpose(1, 2)).transpose(1, 2)[..., :2]
    norm = grid.normalize(src).view(B, grid.ny, grid.nx, 2).to(feat2d.dtype)
    out = F.grid_sample(feat2d, norm, mode="bilinear", padding_mode="zeros", align_corners=False)
    if is_voxel:
        out = out.view(B, C, Z, Y, X)
    return out
