import torch
import torch.nn as nn
import torch.nn.functional as F

from ..registry import BEV_ENCODERS
from .layers import ResBlock, conv_bn_relu


@BEV_ENCODERS.register_module()
class ResBEVEncoder(nn.Module):
    """stride 1/2/4 三层残差，再上采样回 stride 1。输出与输入同分辨率。"""

    def __init__(self, in_channels: int = 128, out_channels: int = 128, num_blocks=(2, 2, 2)):
        super().__init__()
        c = out_channels
        self.stem = conv_bn_relu(in_channels, c)
        self.s1 = nn.Sequential(*[ResBlock(c, c) for _ in range(num_blocks[0])])
        self.s2 = nn.Sequential(ResBlock(c, 2 * c, 2), *[ResBlock(2 * c, 2 * c) for _ in range(num_blocks[1] - 1)])
        self.s4 = nn.Sequential(ResBlock(2 * c, 4 * c, 2), *[ResBlock(4 * c, 4 * c) for _ in range(num_blocks[2] - 1)])
        self.up = conv_bn_relu(c + 2 * c + 4 * c, c)

    def forward(self, x):
        x1 = self.s1(self.stem(x))
        x2 = self.s2(x1)
        x4 = self.s4(x2)
        size = x1.shape[-2:]
        x2u = F.interpolate(x2, size=size, mode="bilinear", align_corners=False)
        x4u = F.interpolate(x4, size=size, mode="bilinear", align_corners=False)
        return self.up(torch.cat([x1, x2u, x4u], 1))


@BEV_ENCODERS.register_module()
class GODVoxelEncoder(nn.Module):
    """近场体素特征：图像抬升得到的体素 + 广播到每层高度的 BEV 上下文（含 Lidar/Radar 与时序）。

    occ、occ flow、SDF 三个头读同一份输出。
    """

    def __init__(self, voxel_channels: int = 32, bev_channels: int = 128, out_channels: int = 32,
                 num_z: int = 13, num_blocks: int = 2):
        super().__init__()
        self.bev_to_voxel = nn.Conv2d(bev_channels, out_channels, 1)
        self.z_embed = nn.Parameter(torch.zeros(1, out_channels, num_z, 1, 1))
        self.stem = conv_bn_relu(voxel_channels + out_channels, out_channels, dim=3)
        self.blocks = nn.Sequential(*[ResBlock(out_channels, out_channels, dim=3) for _ in range(num_blocks)])
        nn.init.normal_(self.z_embed, std=0.02)

    def forward(self, voxel: torch.Tensor, bev: torch.Tensor) -> torch.Tensor:
        Z = voxel.shape[2]
        ctx = self.bev_to_voxel(bev).unsqueeze(2).expand(-1, -1, Z, -1, -1) + self.z_embed
        return self.blocks(self.stem(torch.cat([voxel, ctx.to(voxel.dtype)], 1)))
