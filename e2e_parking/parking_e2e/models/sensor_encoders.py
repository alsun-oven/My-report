from typing import Dict, List, Optional

import torch
import torch.nn as nn

from ..geometry import Grid3D
from ..registry import SENSOR_ENCODERS
from .layers import MLP, ResBlock, conv_bn_relu


@SENSOR_ENCODERS.register_module()
class PillarEncoder(nn.Module):
    """PointPillars 式的柱体编码，点云在自车系。输入 list[B] of (Ni, in_channels)，前 3 维为 xyz。"""

    def __init__(self, grid, in_channels: int = 4, feat_channels: int = 64, out_channels: int = 64,
                 num_blocks: int = 2):
        super().__init__()
        self.grid = Grid3D.from_cfg(grid)
        self.in_channels = in_channels
        # 原始特征 + 相对柱内均值(3) + 相对柱中心(2)
        self.point_net = nn.Sequential(nn.Linear(in_channels + 5, feat_channels), nn.LayerNorm(feat_channels),
                                       nn.ReLU(inplace=True))
        blocks = [conv_bn_relu(feat_channels, out_channels)]
        blocks += [ResBlock(out_channels, out_channels) for _ in range(num_blocks)]
        self.bev_net = nn.Sequential(*blocks)
        self.feat_channels = feat_channels

    def preprocess(self, pts: torch.Tensor) -> torch.Tensor:
        return pts

    def forward(self, points: List[torch.Tensor], batch_size: Optional[int] = None) -> torch.Tensor:
        B = batch_size or len(points)
        ny, nx = self.grid.ny, self.grid.nx
        dev = self.point_net[0].weight.device
        pts_all, bidx_all = [], []
        for b, p in enumerate(points):
            p = self.preprocess(p.to(dev).float()[:, :self.in_channels])
            pts_all.append(p)
            bidx_all.append(torch.full((p.shape[0],), b, device=dev, dtype=torch.long))
        pts = torch.cat(pts_all, 0)
        bidx = torch.cat(bidx_all, 0)

        ix, iy, iz, valid = self.grid.to_index(pts[:, :3])
        pts, bidx, ix, iy = pts[valid], bidx[valid], ix[valid], iy[valid]
        pid = (bidx * ny + iy) * nx + ix
        n_cells = B * ny * nx
        pooled = torch.zeros(n_cells, self.feat_channels, device=dev, dtype=pts.dtype)
        if pts.shape[0] > 0:
            cnt = torch.zeros(n_cells, device=dev, dtype=pts.dtype).index_add_(0, pid, torch.ones_like(pid, dtype=pts.dtype))
            mean = torch.zeros(n_cells, 3, device=dev, dtype=pts.dtype).index_add_(0, pid, pts[:, :3])
            mean = mean / cnt.clamp(min=1)[:, None]
            cx = (ix.to(pts.dtype) + 0.5) * self.grid.xbound[2] + self.grid.xbound[0]
            cy = (iy.to(pts.dtype) + 0.5) * self.grid.ybound[2] + self.grid.ybound[0]
            feat = torch.cat([pts, pts[:, :3] - mean[pid], (pts[:, 0] - cx)[:, None], (pts[:, 1] - cy)[:, None]], 1)
            feat = self.point_net(feat)
            pooled = pooled.to(feat.dtype).scatter_reduce(
                0, pid[:, None].expand(-1, feat.shape[1]), feat, reduce="amax", include_self=False)
        bev = pooled.view(B, ny, nx, -1).permute(0, 3, 1, 2).contiguous()
        return self.bev_net(bev)


@SENSOR_ENCODERS.register_module()
class RadarPillarEncoder(PillarEncoder):
    """毫米波点 (x, y, z, vx_comp, vy_comp, rcs, dt)。速度、RCS 归一化后走同一套柱体编码。

    雷达点稀疏，默认多叠一层残差块扩大感受野。
    """

    def __init__(self, grid, in_channels: int = 7, feat_channels: int = 32, out_channels: int = 32,
                 num_blocks: int = 3, vel_scale: float = 5.0, rcs_scale: float = 20.0):
        super().__init__(grid, in_channels, feat_channels, out_channels, num_blocks)
        self.vel_scale = vel_scale
        self.rcs_scale = rcs_scale

    def preprocess(self, pts):
        pts = pts.clone()
        pts[:, 3:5] = pts[:, 3:5] / self.vel_scale
        if pts.shape[1] > 5:
            pts[:, 5] = pts[:, 5] / self.rcs_scale
        return pts


@SENSOR_ENCODERS.register_module()
class NavigationEncoder(nn.Module):
    """导航与自车状态 -> token 序列。

    输入 nav dict：
      target_slot (B, 4, 2)  目标车位四角，顺序为入口左、入口右、里侧右、里侧左
      target_valid (B,)      寻库阶段可能还没有目标车位
      route (B, P, 2)        停车场内的导航路线点（自车系）
      route_valid (B, P)
      ego_status (B, E)      速度、横摆角速度、方向盘角、加速度、当前档位 one-hot 等
      phase (B,)             0 寻库，1 泊入
    输出 tokens (B, 4 + P + 1, C)、key_padding_mask (True 表示忽略)、global (B, C)。
    """

    def __init__(self, embed_dims: int = 256, ego_dim: int = 8, max_route_pts: int = 32,
                 num_phases: int = 2, coord_scale: float = 12.0):
        super().__init__()
        self.coord_scale = coord_scale
        self.slot_mlp = MLP(2, embed_dims, embed_dims, 2)
        self.corner_emb = nn.Embedding(4, embed_dims)
        self.route_mlp = MLP(2, embed_dims, embed_dims, 2)
        self.route_pos = nn.Embedding(max_route_pts, embed_dims)
        self.ego_mlp = MLP(ego_dim, embed_dims, embed_dims, 2)
        self.phase_emb = nn.Embedding(num_phases, embed_dims)
        self.out_norm = nn.LayerNorm(embed_dims)

    def forward(self, nav: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        slot = self.slot_mlp(nav["target_slot"].float() / self.coord_scale) + self.corner_emb.weight[None]
        route = nav["route"].float()
        P = route.shape[1]
        route_tok = self.route_mlp(route / self.coord_scale) + self.route_pos.weight[None, :P]
        ego_tok = self.ego_mlp(nav["ego_status"].float()) + self.phase_emb(nav["phase"].long())
        tokens = self.out_norm(torch.cat([slot, route_tok, ego_tok[:, None]], 1))

        B = slot.shape[0]
        slot_pad = ~nav["target_valid"].bool()[:, None].expand(B, 4)
        route_pad = ~nav["route_valid"].bool()
        ego_pad = torch.zeros(B, 1, dtype=torch.bool, device=slot.device)
        mask = torch.cat([slot_pad, route_pad, ego_pad], 1)

        w = (~slot_pad).float()[..., None]
        slot_mean = (tokens[:, :4] * w).sum(1) / w.sum(1).clamp(min=1)
        return dict(tokens=tokens, mask=mask, global_feat=tokens[:, -1] + slot_mean)
