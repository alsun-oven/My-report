from typing import List, Optional, Sequence

import torch
import torch.nn as nn

from ..geometry import Grid3D, warp_bev
from ..registry import FUSERS
from .layers import ResBlock, conv_bn_relu


@FUSERS.register_module()
class ConvFuser(nn.Module):
    """相机 / Lidar / Radar 的 BEV 拼接融合。缺失的模态补零；训练时按概率随机丢掉非相机模态，
    让模型在 Lidar 或 Radar 掉线时还能工作。
    """

    def __init__(self, in_channels: Sequence[int], out_channels: int = 128,
                 modality_dropout: Sequence[float] = (0.0, 0.1, 0.1)):
        super().__init__()
        self.in_channels = list(in_channels)
        self.modality_dropout = list(modality_dropout)
        self.fuse = nn.Sequential(conv_bn_relu(sum(in_channels), out_channels), ResBlock(out_channels, out_channels))

    def forward(self, feats: List[Optional[torch.Tensor]]) -> torch.Tensor:
        ref = next(f for f in feats if f is not None)
        B, _, H, W = ref.shape
        xs = []
        for f, c, p in zip(feats, self.in_channels, self.modality_dropout):
            if f is None:
                f = ref.new_zeros(B, c, H, W)
            elif self.training and p > 0:
                keep = (torch.rand(B, 1, 1, 1, device=f.device) >= p).to(f.dtype)
                f = f * keep
            xs.append(f)
        return self.fuse(torch.cat(xs, 1))


@FUSERS.register_module()
class TemporalAlignFuser(nn.Module):
    """把历史帧的 BEV 或体素特征按自车位姿对齐到当前帧后拼接融合。

    hist: list[T]，T_cur_from_hist (B, T, 4, 4)，hist_valid (B, T)。缺历史帧时补零，保持通道数固定。
    """

    def __init__(self, grid, channels: int, num_history: int, dim: int = 2):
        super().__init__()
        self.grid = Grid3D.from_cfg(grid)
        self.num_history = num_history
        self.dim = dim
        self.fuse = nn.Sequential(conv_bn_relu(channels * (num_history + 1), channels, dim=dim),
                                  ResBlock(channels, channels, dim=dim))

    def forward(self, cur: torch.Tensor, hist: Optional[List[torch.Tensor]] = None,
                T_cur_from_hist: Optional[torch.Tensor] = None,
                hist_valid: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.num_history == 0:
            return cur
        feats = [cur]
        hist = hist or []
        for t in range(self.num_history):
            if t < len(hist) and T_cur_from_hist is not None:
                T_hist_from_cur = torch.linalg.inv(T_cur_from_hist[:, t].float())
                w = warp_bev(hist[t], T_hist_from_cur, self.grid).to(cur.dtype)
                if hist_valid is not None:
                    w = w * hist_valid[:, t].to(cur.dtype).view(-1, *([1] * (w.dim() - 1)))
            else:
                w = torch.zeros_like(cur)
            feats.append(w)
        return self.fuse(torch.cat(feats, 1))
