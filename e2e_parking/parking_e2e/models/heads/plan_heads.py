"""规划两个查询组：车位查询（多边形 + 空闲/占用）和轨迹（DiT + Flow Matching，档位单独分类）。

两组查询只读 BEV 特征、GOD 体素特征和导航 token，不读检测框、分割图和 PV 解码。
碰撞项在训练时沿 Flow Matching 的一步估计轨迹采样 SDF；推理时 SDF 作为拒绝器从 K 条候选里选。
"""
import math
from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...geometry import Grid3D
from ...registry import HEADS, LOSSES, MATCHERS
from ..layers import MLP
from .base import BaseHead
from .bev_heads import BEVQueryDecoder


@HEADS.register_module()
class SlotHead(BaseHead):
    """车位查询。四角顺序：入口左、入口右、里侧右、里侧左（米，自车系）。

    gt_slots: list[B] of (S, 4, 2)；gt_slot_occupied: list[B] of (S,)。
    输出的 tokens (B, Q, plan_dims) 给轨迹头做交叉注意力。
    """

    def __init__(self, grid, in_channels: int = 128, embed_dims: int = 256, plan_dims: int = 256,
                 num_queries: int = 32, num_layers: int = 3, num_heads: int = 8, num_points: int = 4,
                 corner_scale: float = 4.0, matcher=dict(type="SlotMatcher"),
                 loss_exist=dict(type="SigmoidFocalLoss"), corner_weight: float = 1.0,
                 occupied_weight: float = 0.5, score_thr: float = 0.4):
        super().__init__()
        self.grid = Grid3D.from_cfg(grid)
        self.decoder = BEVQueryDecoder(in_channels, embed_dims, num_queries, num_layers, num_heads, num_points)
        L = num_layers
        self.exist_branches = nn.ModuleList([nn.Linear(embed_dims, 1) for _ in range(L)])
        self.corner_branches = nn.ModuleList([MLP(embed_dims, embed_dims, 8, 2) for _ in range(L)])
        self.occ_branches = nn.ModuleList([nn.Linear(embed_dims, 1) for _ in range(L)])
        for br in self.exist_branches:
            nn.init.constant_(br.bias, -4.6)
        self.token_proj = nn.Linear(embed_dims, plan_dims)
        self.corner_scale = corner_scale
        self.matcher = MATCHERS.build(matcher)
        self.loss_exist = LOSSES.build(loss_exist)
        self.corner_weight = corner_weight
        self.occupied_weight = occupied_weight
        self.score_thr = score_thr

    def _center_m(self, c):
        lo = c.new_tensor(self.grid.lower[:2])
        hi = c.new_tensor(self.grid.upper[:2])
        return c * (hi - lo) + lo

    def forward(self, feats):
        hs, centers = self.decoder(feats["bev"])
        out = dict(exist=[], corners=[], occupied=[])
        for l, (q, c) in enumerate(zip(hs, centers)):
            out["exist"].append(self.exist_branches[l](q)[..., 0])
            off = self.corner_branches[l](q).view(*q.shape[:2], 4, 2) * self.corner_scale
            out["corners"].append(self._center_m(c)[:, :, None, :] + off)
            out["occupied"].append(self.occ_branches[l](q)[..., 0])
        out["tokens"] = self.token_proj(hs[-1])
        return out

    def loss(self, preds, feats, batch):
        res = {}
        B = preds["exist"][0].shape[0]
        dev = preds["exist"][0].device
        gts = [(batch["gt_slots"][b].to(dev).float(), batch["gt_slot_occupied"][b].to(dev).float()) for b in range(B)]
        num_gt = max(sum(g[0].shape[0] for g in gts), 1)
        for l in range(len(preds["exist"])):
            ex, co, oc = preds["exist"][l], preds["corners"][l], preds["occupied"][l]
            ex_t = torch.zeros_like(ex)
            c_losses, o_losses = [], []
            for b, (slots, occ) in enumerate(gts):
                qi, gi = self.matcher(ex[b], co[b], slots)
                if qi.numel() == 0:
                    continue
                qi, gi = qi.to(dev), gi.to(dev)
                ex_t[b, qi] = 1.0
                c_losses.append((co[b, qi] - slots[gi]).abs().sum((1, 2)))
                o_losses.append(F.binary_cross_entropy_with_logits(oc[b, qi].float(), occ[gi], reduction="none"))
            res[f"loss_slot_exist_d{l}"] = self.loss_exist(ex, ex_t, avg_factor=num_gt)
            zero = co.sum() * 0
            res[f"loss_slot_corner_d{l}"] = self.corner_weight * torch.cat(c_losses).sum() / num_gt / 8 if c_losses else zero
            res[f"loss_slot_occ_d{l}"] = self.occupied_weight * torch.cat(o_losses).mean() if o_losses else zero
        return res

    @torch.no_grad()
    def predict(self, preds, feats, batch):
        ex = preds["exist"][-1].sigmoid()
        res = []
        for b in range(ex.shape[0]):
            keep = ex[b] > self.score_thr
            res.append(dict(corners=preds["corners"][-1][b][keep], scores=ex[b][keep],
                            occupied=preds["occupied"][-1][b][keep].sigmoid()))
        return dict(slots=res)


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class TimestepEmbed(nn.Module):
    def __init__(self, d, freq_dim=256):
        super().__init__()
        self.freq_dim = freq_dim
        self.mlp = nn.Sequential(nn.Linear(freq_dim, d), nn.SiLU(), nn.Linear(d, d))

    def forward(self, tau):
        half = self.freq_dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=tau.device, dtype=torch.float32) / half)
        args = (tau.float() * 1000)[:, None] * freqs[None]
        return self.mlp(torch.cat([torch.cos(args), torch.sin(args)], dim=-1))


class DiTBlock(nn.Module):
    """自注意力（轨迹点之间）→ 交叉注意力（场景 token）→ FFN，每步 adaLN-Zero 调制。"""

    def __init__(self, d, heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.self_attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.cross_attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.norm3 = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(d, 9 * d))
        nn.init.zeros_(self.adaLN[-1].weight)
        nn.init.zeros_(self.adaLN[-1].bias)

    def forward(self, x, ctx, c, ctx_mask=None):
        s1, b1, g1, s2, b2, g2, s3, b3, g3 = self.adaLN(c).unsqueeze(1).chunk(9, dim=-1)
        h = modulate(self.norm1(x), b1, s1)
        x = x + g1 * self.self_attn(h, h, h, need_weights=False)[0]
        h = modulate(self.norm2(x), b2, s2)
        x = x + g2 * self.cross_attn(h, ctx, ctx, key_padding_mask=ctx_mask, need_weights=False)[0]
        h = modulate(self.norm3(x), b3, s3)
        return x + g3 * self.mlp(h)


def vehicle_footprint(length, width, rear_overhang, heights, spacing):
    """车身外轮廓采样点（后轴中心为原点）与部位标签 0 前保杠 / 1 后保杠 / 2 侧面。"""
    xf, xr, yl = length - rear_overhang, -rear_overhang, width / 2
    pts, tags = [], []
    nx = max(int(round(length / spacing)), 2)
    ny = max(int(round(width / spacing)), 2)
    for z in heights:
        for i in range(ny + 1):
            y = -yl + width * i / ny
            pts += [(xf, y, z), (xr, y, z)]
            tags += [0, 1]
        for i in range(1, nx):
            x = xr + length * i / nx
            pts += [(x, yl, z), (x, -yl, z)]
            tags += [2, 2]
    return torch.tensor(pts, dtype=torch.float32), torch.tensor(tags, dtype=torch.long)


@HEADS.register_module()
class FlowMatchingTrajHead(BaseHead):
    """轨迹头。traj 每点 (x, y, cos_yaw, sin_yaw)，档位 0 前进 / 1 倒车 / 2 停车单独分类。

    gt_traj (B, T, 4)、gt_gear (B, T)、gt_traj_mask (B, T)。
    条件：BEV token（池化）、GOD token（池化）、车位查询 token、导航 token；
    时间步和导航全局向量（含自车状态、阶段）走 adaLN。
    """

    def __init__(self, grid, bev_channels: int = 128, voxel_channels: int = 32, embed_dims: int = 256,
                 num_heads: int = 8, depth: int = 3, num_pts: int = 40,
                 bev_tokens_hw: Sequence[int] = (24, 24), god_tokens_zyx: Sequence[int] = (2, 12, 12),
                 traj_scale: Sequence[float] = (10.0, 10.0, 1.0, 1.0), num_gears: int = 3, gear_layers: int = 2,
                 switch_weight: float = 4.0, gear_weight: float = 0.5,
                 num_samples: int = 8, sample_steps: int = 8,
                 collision_weight: float = 0.1, vehicle: Optional[Dict] = None,
                 clearance: Sequence[float] = (0.30, 0.30, 0.20), use_slot_tokens: bool = True):
        super().__init__()
        self.grid = Grid3D.from_cfg(grid)
        d = embed_dims
        self.num_pts = num_pts
        self.bev_hw = tuple(bev_tokens_hw)
        self.god_zyx = tuple(god_tokens_zyx)
        self.bev_proj = nn.Linear(bev_channels, d)
        self.god_proj = nn.Linear(voxel_channels, d)
        self.bev_pos = nn.Parameter(torch.zeros(1, self.bev_hw[0] * self.bev_hw[1], d))
        self.god_pos = nn.Parameter(torch.zeros(1, self.god_zyx[0] * self.god_zyx[1] * self.god_zyx[2], d))
        self.type_emb = nn.Embedding(4, d)  # 0 bev, 1 god, 2 slot, 3 nav
        self.use_slot_tokens = use_slot_tokens
        nn.init.normal_(self.bev_pos, std=0.02)
        nn.init.normal_(self.god_pos, std=0.02)

        self.traj_in = nn.Linear(4, d)
        self.traj_pos = nn.Parameter(torch.zeros(1, num_pts, d))
        nn.init.normal_(self.traj_pos, std=0.02)
        self.t_emb = TimestepEmbed(d)
        self.cond_proj = nn.Linear(d, d)
        self.blocks = nn.ModuleList([DiTBlock(d, num_heads) for _ in range(depth)])
        self.final_norm = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.final_adaLN = nn.Sequential(nn.SiLU(), nn.Linear(d, 2 * d))
        self.out = nn.Linear(d, 4)
        for m in (self.final_adaLN[-1], self.out):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

        self.gear_queries = nn.Parameter(torch.zeros(1, num_pts, d))
        nn.init.normal_(self.gear_queries, std=0.02)
        layer = nn.TransformerDecoderLayer(d, num_heads, 4 * d, dropout=0.0, batch_first=True)
        self.gear_decoder = nn.TransformerDecoder(layer, gear_layers)
        self.gear_out = nn.Linear(d, num_gears)

        self.register_buffer("traj_scale", torch.tensor(traj_scale, dtype=torch.float32), persistent=False)
        self.switch_weight = switch_weight
        self.gear_weight = gear_weight
        self.num_samples = num_samples
        self.sample_steps = sample_steps
        self.collision_weight = collision_weight
        vcfg = dict(length=4.9, width=1.9, rear_overhang=1.0, heights=(0.3, 0.8, 1.4), spacing=0.25)
        vcfg.update(vehicle or {})
        fp, tags = vehicle_footprint(**vcfg)
        self.register_buffer("footprint", fp, persistent=False)
        self.register_buffer("fp_margin", torch.tensor(clearance, dtype=torch.float32)[tags], persistent=False)

    # ------------------------------------------------------------------ context
    def build_context(self, feats):
        bev = F.adaptive_avg_pool2d(feats["bev"], self.bev_hw).flatten(2).transpose(1, 2)
        god = F.adaptive_avg_pool3d(feats["voxel"], self.god_zyx).flatten(2).transpose(1, 2)
        toks = [self.bev_proj(bev) + self.bev_pos + self.type_emb.weight[0],
                self.god_proj(god) + self.god_pos + self.type_emb.weight[1]]
        B = bev.shape[0]
        masks = [torch.zeros(B, toks[0].shape[1] + toks[1].shape[1], dtype=torch.bool, device=bev.device)]
        slot = feats.get("preds", {}).get("slot")
        if self.use_slot_tokens and slot is not None:
            toks.append(slot["tokens"] + self.type_emb.weight[2])
            masks.append(torch.zeros(B, slot["tokens"].shape[1], dtype=torch.bool, device=bev.device))
        nav = feats["nav"]
        toks.append(nav["tokens"] + self.type_emb.weight[3])
        masks.append(nav["mask"])
        return torch.cat(toks, 1), torch.cat(masks, 1), nav["global_feat"]

    def velocity(self, x_t, tau, ctx, ctx_mask, g):
        c = self.t_emb(tau).to(g.dtype) + self.cond_proj(g)
        x = self.traj_in(x_t) + self.traj_pos
        for blk in self.blocks:
            x = blk(x, ctx, c, ctx_mask)
        shift, scale = self.final_adaLN(c).unsqueeze(1).chunk(2, dim=-1)
        return self.out(modulate(self.final_norm(x), shift, scale))

    def gear_logits(self, ctx, ctx_mask):
        q = self.gear_queries.expand(ctx.shape[0], -1, -1)
        return self.gear_out(self.gear_decoder(q, ctx, memory_key_padding_mask=ctx_mask))

    def forward(self, feats):
        ctx, mask, g = self.build_context(feats)
        return dict(ctx=ctx, ctx_mask=mask, global_feat=g, gear_logits=self.gear_logits(ctx, mask))

    # ------------------------------------------------------------------ geometry
    def normalize(self, traj):
        return traj / self.traj_scale

    def denormalize(self, x):
        return x * self.traj_scale

    def footprint_points(self, traj):
        """traj (B, T, 4) 米 -> (B, T, P, 3) 车身轮廓点（自车系）。"""
        yaw = torch.atan2(traj[..., 3], traj[..., 2])
        c, s = yaw.cos()[..., None], yaw.sin()[..., None]
        fx, fy, fz = self.footprint.unbind(-1)
        x = traj[..., 0:1] + c * fx - s * fy
        y = traj[..., 1:2] + s * fx + c * fy
        z = fz.expand_as(x)
        return torch.stack([x, y, z], -1)

    def _in_grid(self, pts):
        lo = pts.new_tensor(self.grid.lower)
        hi = pts.new_tensor(self.grid.upper)
        return ((pts >= lo) & (pts < hi)).all(-1)

    def clearance_slack(self, traj, sdf_fn):
        """每个轮廓点的 sdf - 部位间隙阈值，(B, T, P)。网格外的点视为不受约束。"""
        pts = self.footprint_points(traj)
        B, T, P, _ = pts.shape
        sdf = sdf_fn(pts.view(B, T * P, 3)).view(B, T, P)
        slack = sdf - self.fp_margin
        return torch.where(self._in_grid(pts), slack, torch.full_like(slack, 1e3))

    # ------------------------------------------------------------------ train
    def loss(self, preds, feats, batch):
        gt = batch["gt_traj"].float()
        mask = batch["gt_traj_mask"].bool()
        ctx, ctx_mask, g = preds["ctx"], preds["ctx_mask"], preds["global_feat"]
        B = gt.shape[0]
        x1 = self.normalize(gt)
        tau = torch.rand(B, device=gt.device)
        x0 = torch.randn_like(x1)
        t = tau[:, None, None]
        x_t = (1 - t) * x0 + t * x1
        v = self.velocity(x_t, tau, ctx, ctx_mask, g).float()
        m = mask[..., None].float()
        res = dict(loss_plan_fm=(((v - (x1 - x0)) ** 2) * m).sum() / (m.sum() * 4).clamp(min=1))

        gear = batch["gt_gear"].long()
        gl = preds["gear_logits"].float()
        switch = torch.zeros_like(gl[..., 0])
        switch[:, 1:] = (gear[:, 1:] != gear[:, :-1]).float()
        switch = torch.maximum(switch, torch.maximum(F.pad(switch[:, 1:], (0, 1)), F.pad(switch[:, :-1], (1, 0))))
        w = (1.0 + self.switch_weight * switch) * mask.float()
        ce = F.cross_entropy(gl.flatten(0, 1), gear.clamp(min=0).flatten(), reduction="none").view_as(w)
        res["loss_plan_gear"] = self.gear_weight * (ce * w).sum() / w.sum().clamp(min=1)

        sdf_fn = feats.get("sdf_fn")
        if sdf_fn is not None and self.collision_weight > 0:
            x1_hat = self.denormalize(x_t + (1 - t) * v)
            slack = self.clearance_slack(x1_hat, sdf_fn)
            pen = F.relu(-slack) * mask[..., None].float() * tau[:, None, None]
            res["loss_plan_collision"] = self.collision_weight * pen.sum() / (mask.sum() * slack.shape[-1]).clamp(min=1)

        with torch.no_grad():
            res["gear_acc"] = ((gl.argmax(-1) == gear) & mask).sum() / mask.sum().clamp(min=1)
        return res

    # ------------------------------------------------------------------ inference
    @torch.no_grad()
    def sample(self, ctx, ctx_mask, g, num_samples, steps):
        B = ctx.shape[0]
        K = num_samples
        ctx_k = ctx.repeat_interleave(K, 0)
        mask_k = ctx_mask.repeat_interleave(K, 0)
        g_k = g.repeat_interleave(K, 0)
        x = torch.randn(B * K, self.num_pts, 4, device=ctx.device)
        dt = 1.0 / steps
        for i in range(steps):
            tau = torch.full((B * K,), i * dt, device=ctx.device)
            x = x + dt * self.velocity(x, tau, ctx_k, mask_k, g_k).float()
        traj = self.denormalize(x).view(B, K, self.num_pts, 4)
        traj[..., 2:4] = F.normalize(traj[..., 2:4], dim=-1)
        return traj

    @torch.no_grad()
    def predict(self, preds, feats, batch):
        ctx, ctx_mask, g = preds["ctx"], preds["ctx_mask"], preds["global_feat"]
        cands = self.sample(ctx, ctx_mask, g, self.num_samples, self.sample_steps)   # (B, K, T, 4)
        B, K, T, _ = cands.shape
        ade = (cands[:, :, None, :, :2] - cands[:, None, :, :, :2]).norm(dim=-1).mean(-1)  # (B, K, K)
        sdf_fn = feats.get("sdf_fn")
        if sdf_fn is not None:
            # sdf_fn 的体素特征 batch 为 B，候选沿时间维摊开查询
            slack = self.clearance_slack(cands.view(B, K * T, 4), sdf_fn).view(B, K, -1).amin(-1)
            safe = slack >= 0
        else:
            slack = torch.full((B, K), float("inf"), device=cands.device)
            safe = torch.ones(B, K, dtype=torch.bool, device=cands.device)
        # 安全候选里取 medoid；全部不安全时取间隙最大的那条，并标记 plan_safe=False
        big = ade.max() + 1.0
        ade_safe = torch.where(safe[:, None, :], ade, big)
        score = torch.where(safe, ade_safe.mean(-1), torch.full_like(slack, float("inf")))
        pick = torch.where(safe.any(1), score.argmin(1), slack.argmax(1))
        bi = torch.arange(B, device=cands.device)
        traj = cands[bi, pick]
        yaw = torch.atan2(traj[..., 3], traj[..., 2])
        return dict(traj=torch.cat([traj[..., :2], yaw[..., None]], -1),
                    gear=preds["gear_logits"].argmax(-1),
                    plan_min_slack=slack[bi, pick], plan_safe=safe[bi, pick],
                    traj_candidates=cands)
