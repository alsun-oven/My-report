from typing import List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..registry import BACKBONES, NECKS
from .layers import conv_bn_relu


@BACKBONES.register_module()
class TorchvisionResNet(nn.Module):
    """输出 stride 8/16/32 三层。7 路针孔与 4 路鱼眼共用这一份骨干。"""

    def __init__(self, depth: int = 18, weights: Optional[str] = None,
                 out_indices: Sequence[int] = (2, 3, 4), frozen_stages: int = -1):
        super().__init__()
        import torchvision
        net = getattr(torchvision.models, f"resnet{depth}")(weights=weights)
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.stages = nn.ModuleList([net.layer1, net.layer2, net.layer3, net.layer4])
        self.out_indices = tuple(out_indices)
        expansion = 1 if depth < 50 else 4
        self.out_channels = [64 * expansion * 2 ** (i - 1) for i in self.out_indices]
        self.frozen_stages = frozen_stages
        self._freeze()

    def _freeze(self):
        if self.frozen_stages >= 0:
            for p in self.stem.parameters():
                p.requires_grad = False
        for i in range(self.frozen_stages):
            for p in self.stages[i].parameters():
                p.requires_grad = False

    def forward(self, x) -> List[torch.Tensor]:
        x = self.stem(x)
        outs = []
        for i, stage in enumerate(self.stages, start=1):
            x = stage(x)
            if i in self.out_indices:
                outs.append(x)
        return outs


@NECKS.register_module()
class SimpleFPN(nn.Module):
    """自顶向下融合，按 out_strides 输出。默认给 PV 头 stride 8，给 LSS stride 16。"""

    def __init__(self, in_channels: Sequence[int], out_channels: int = 256,
                 in_strides: Sequence[int] = (8, 16, 32), out_strides: Sequence[int] = (8, 16)):
        super().__init__()
        self.lateral = nn.ModuleList([nn.Conv2d(c, out_channels, 1) for c in in_channels])
        self.smooth = nn.ModuleList([conv_bn_relu(out_channels, out_channels) for _ in in_channels])
        self.in_strides = list(in_strides)
        self.out_ids = [self.in_strides.index(s) for s in out_strides]

    def forward(self, feats: List[torch.Tensor]) -> List[torch.Tensor]:
        lat = [l(f) for l, f in zip(self.lateral, feats)]
        for i in range(len(lat) - 1, 0, -1):
            lat[i - 1] = lat[i - 1] + F.interpolate(lat[i], size=lat[i - 1].shape[-2:], mode="bilinear",
                                                    align_corners=False)
        return [self.smooth[i](lat[i]) for i in self.out_ids]
