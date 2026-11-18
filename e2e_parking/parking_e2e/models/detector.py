"""端到端泊车模型。

计算图：
  7 针孔 + 4 鱼眼 ──共享骨干/FPN──┬─ stride 8 ─> PV 头（重建 / 图像分割 / 图像检测）
                                  └─ stride 16 ─> 多组 LSS ─> 相机体素 ─┬─> 相机 BEV
  Lidar ─ 柱体编码 ─> Lidar BEV ─┐                                      │
  Radar ─ 柱体编码 ─> Radar BEV ─┴──────────── ConvFuser <──────────────┘
                                                 │ 时序对齐（历史帧按自车位姿 warp）
                                                 v
                                            BEV 编码 ──> BEV 头（检测 / 分割）
                                                 │
                  相机体素（时序对齐）+ BEV 上下文 ──> GOD 体素 ──> occ / occ flow / SDF
  导航 ─> 导航 token ─┐
  BEV + GOD + 车位 token ─> 车位查询、DiT 轨迹（碰撞项采样 SDF）
"""
from typing import Any, Dict, Optional, Sequence

import torch
import torch.nn as nn

from ..geometry import Grid3D
from ..registry import (BACKBONES, BEV_ENCODERS, FUSERS, HEADS, MODELS, NECKS, SENSOR_ENCODERS,
                        VIEW_TRANSFORMS)
from .layers import build_with_defaults

DEFAULT_LOSS_WEIGHTS = dict(
    traj=1.0, slot=1.0,
    bev_det=0.5, bev_seg=0.5,
    occ=0.4, occ_flow=0.2, sdf=0.3,
    pv_det=0.2, pv_seg=0.2, pv_depth=0.2,
)


@MODELS.register_module()
class ParkingE2E(nn.Module):
    def __init__(self, grid, cam_groups: Dict[str, dict], img_backbone: dict, img_neck: dict,
                 view_transform: dict, nav_encoder: dict, fuser: dict, bev_encoder: dict, god_encoder: dict,
                 lidar_encoder: Optional[dict] = None, radar_encoder: Optional[dict] = None,
                 bev_temporal: Optional[dict] = None, voxel_temporal: Optional[dict] = None,
                 heads: Optional[Dict[str, dict]] = None, loss_weights: Optional[Dict[str, float]] = None,
                 deploy_heads: Optional[Sequence[str]] = None):
        super().__init__()
        self.grid = Grid3D.from_cfg(grid)
        self.cam_groups = list(cam_groups)
        g = self.grid.to_dict()

        self.img_backbone = BACKBONES.build(img_backbone)
        self.img_neck = build_with_defaults(NECKS, img_neck, in_channels=self.img_backbone.out_channels)
        self.view_transform = build_with_defaults(VIEW_TRANSFORMS, view_transform, grid=g, cam_groups=cam_groups)
        self.lidar_encoder = build_with_defaults(SENSOR_ENCODERS, lidar_encoder, grid=g)
        self.radar_encoder = build_with_defaults(SENSOR_ENCODERS, radar_encoder, grid=g)
        self.nav_encoder = SENSOR_ENCODERS.build(nav_encoder)
        self.fuser = FUSERS.build(fuser)
        self.bev_temporal = build_with_defaults(FUSERS, bev_temporal, grid=g)
        self.voxel_temporal = build_with_defaults(FUSERS, voxel_temporal, grid=g)
        self.bev_encoder = BEV_ENCODERS.build(bev_encoder)
        self.god_encoder = build_with_defaults(BEV_ENCODERS, god_encoder, num_z=self.grid.nz)

        heads = dict(heads or {})
        if "traj" in heads:  # 轨迹头读车位头的 token，必须排在最后
            heads["traj"] = heads.pop("traj")
        self.heads = nn.ModuleDict({k: build_with_defaults(HEADS, v, grid=g) for k, v in heads.items()})
        self.loss_weights = dict(DEFAULT_LOSS_WEIGHTS)
        self.loss_weights.update(loss_weights or {})
        self.deploy_heads = list(deploy_heads) if deploy_heads is not None else list(self.heads)

    def set_loss_weights(self, weights: Dict[str, float]):
        self.loss_weights.update(weights)

    # ------------------------------------------------------------------ features
    def extract_frame(self, frame: Dict[str, Any]) -> Dict[str, Any]:
        pv, lss_in, shapes, calibs = {}, {}, {}, {}
        for g in self.cam_groups:
            key = f"img_{g}"
            if key not in frame:
                continue
            imgs = frame[key]
            B, N = imgs.shape[:2]
            outs = self.img_neck(self.img_backbone(imgs.flatten(0, 1)))
            pv[g] = outs[0].unflatten(0, (B, N))
            lss_in[g] = outs[1].unflatten(0, (B, N))
            shapes[g] = tuple(imgs.shape[-2:])
            calibs[g] = frame[f"calib_{g}"]
        vt = self.view_transform(lss_in, calibs, shapes)
        B = vt["bev"].shape[0]
        lidar = radar = None
        if self.lidar_encoder is not None and frame.get("lidar_points") is not None:
            lidar = self.lidar_encoder(frame["lidar_points"], B)
        if self.radar_encoder is not None and frame.get("radar_points") is not None:
            radar = self.radar_encoder(frame["radar_points"], B)
        bev = self.fuser([vt["bev"], lidar, radar])
        return dict(pv=pv, depth_logits=vt["depth_logits"], bev=bev, voxel=vt["voxel"])

    def depth_meta(self) -> Dict[str, dict]:
        vt = self.view_transform
        meta = {}
        for g in self.cam_groups:
            dmin, step, n = vt.depth_cfg[g]
            meta[g] = dict(dmin=dmin, step=step, num=n, stride=vt.strides[g], bins=vt.depth_bins(g),
                           depth_type=vt.cam_models[g].depth_type)
        return meta

    def extract_feats(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        cur = self.extract_frame(batch)
        bev, voxel = cur["bev"], cur["voxel"]
        if self.bev_temporal is not None or self.voxel_temporal is not None:
            hist_bev, hist_vox = [], []
            if batch.get("history"):
                with torch.no_grad():
                    for h in batch["history"]:
                        f = self.extract_frame(h)
                        hist_bev.append(f["bev"])
                        hist_vox.append(f["voxel"])
            T, valid = batch.get("T_cur_from_hist"), batch.get("hist_valid")
            if self.bev_temporal is not None:
                bev = self.bev_temporal(bev, hist_bev, T, valid)
            if self.voxel_temporal is not None:
                voxel = self.voxel_temporal(voxel, hist_vox, T, valid)
        bev = self.bev_encoder(bev)
        voxel = self.god_encoder(voxel, bev)
        feats = dict(pv=cur["pv"], depth_logits=cur["depth_logits"], depth_meta=self.depth_meta(),
                     bev=bev, voxel=voxel, nav=self.nav_encoder(batch["nav"]), preds={})
        if "sdf" in self.heads:
            sdf_head = self.heads["sdf"]
            feats["sdf_fn"] = lambda pts: sdf_head.query(voxel, pts, frozen=True)
        return feats

    # ------------------------------------------------------------------ api
    def forward(self, batch: Dict[str, Any], mode: str = "loss"):
        if mode == "loss":
            return self.loss(batch)
        if mode == "predict":
            return self.predict(batch)
        raise ValueError(mode)

    def loss(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        feats = self.extract_feats(batch)
        losses: Dict[str, torch.Tensor] = {}
        for name, head in self.heads.items():
            w = self.loss_weights.get(name, 1.0)
            needed_by_traj = name == "slot" and "traj" in self.heads and self.loss_weights.get("traj", 1.0) > 0
            if w <= 0 and not needed_by_traj:
                continue
            preds = head(feats)
            feats["preds"][name] = preds
            if w <= 0:
                continue
            for k, v in head.loss(preds, feats, batch).items():
                if k.startswith("loss"):
                    losses[f"{k}"] = v * w
                else:
                    losses[f"{name}/{k}"] = v.detach() if torch.is_tensor(v) else v
        losses["loss"] = sum(v for k, v in losses.items() if k.startswith("loss"))
        return losses

    @torch.no_grad()
    def predict(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        feats = self.extract_feats(batch)
        results: Dict[str, Any] = {}
        run = set(self.deploy_heads)
        if "traj" in run and "slot" in self.heads:
            run.add("slot")
        for name, head in self.heads.items():
            if name not in run:
                continue
            preds = head(feats)
            feats["preds"][name] = preds
            if name in self.deploy_heads:
                results.update(head.predict(preds, feats, batch))
        return results
