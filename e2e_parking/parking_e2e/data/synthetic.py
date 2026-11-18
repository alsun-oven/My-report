"""形状与真实数据一致的随机数据，用于冒烟测试整条训练 / 推理链路。标签没有物理意义。"""
import math
from typing import Dict, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from ..geometry import Grid3D
from ..registry import DATASETS


def cam2ego_matrix(x, y, z, yaw, pitch_down=0.0):
    """相机系 x 右、y 下、z 光轴。yaw 为光轴在自车系的朝向，pitch_down 为向下俯角（弧度）。"""
    cy, sy, cp, sp = math.cos(yaw), math.sin(yaw), math.cos(pitch_down), math.sin(pitch_down)
    z_axis = np.array([cy * cp, sy * cp, -sp])
    x_axis = np.array([sy, -cy, 0.0])
    y_axis = np.cross(z_axis, x_axis)
    T = np.eye(4, dtype=np.float32)
    T[:3, 0], T[:3, 1], T[:3, 2] = x_axis, y_axis, z_axis
    T[:3, 3] = (x, y, z)
    return T


# 7 路针孔：前广角、前窄角、左前、右前、左后、右后、后
PINHOLE_RIG = [(1.9, 0.0, 1.4, 0.0, 0.05), (1.9, 0.0, 1.4, 0.0, 0.0), (2.0, 0.9, 1.0, 1.0, 0.1),
               (2.0, -0.9, 1.0, -1.0, 0.1), (1.0, 0.9, 1.0, 2.3, 0.1), (1.0, -0.9, 1.0, -2.3, 0.1),
               (-1.0, 0.0, 1.0, math.pi, 0.1)]
# 4 路鱼眼：前、左、后、右（左右在后视镜下，俯角更大）
FISHEYE_RIG = [(3.7, 0.0, 0.6, 0.0, 0.4), (2.0, 1.0, 1.0, math.pi / 2, 0.9),
               (-1.0, 0.0, 0.9, math.pi, 0.4), (2.0, -1.0, 1.0, -math.pi / 2, 0.9)]


@DATASETS.register_module()
class SyntheticParkingDataset(Dataset):
    def __init__(self, grid, length: int = 8, img_sizes: Optional[Dict[str, Sequence[int]]] = None,
                 bev_seg_upsample: int = 4, num_bev_seg_classes: int = 8, num_pv_seg_classes: int = 8,
                 num_pv_det_classes: int = 4, num_det_classes: int = 7, num_occ_sem: int = 4,
                 num_traj_pts: int = 40, route_pts: int = 32, ego_dim: int = 8,
                 num_occ_pts: int = 4096, num_sdf_pts: int = 4096, num_flow_pts: int = 1024,
                 num_history: int = 0, with_lidar: bool = True, with_radar: bool = True, seed: int = 0):
        self.grid = Grid3D.from_cfg(grid)
        self.length = length
        self.img_sizes = img_sizes or dict(pinhole=(256, 448), fisheye=(320, 512))
        self.up = bev_seg_upsample
        self.k_bev, self.k_pvseg, self.k_pvdet, self.k_det = num_bev_seg_classes, num_pv_seg_classes, num_pv_det_classes, num_det_classes
        self.k_occ_sem = num_occ_sem
        self.T, self.P, self.E = num_traj_pts, route_pts, ego_dim
        self.n_occ, self.n_sdf, self.n_flow = num_occ_pts, num_sdf_pts, num_flow_pts
        self.num_history = num_history
        self.with_lidar, self.with_radar = with_lidar, with_radar
        self.seed = seed

    def __len__(self):
        return self.length

    def _calib(self, group, H, W):
        if group == "pinhole":
            rig = PINHOLE_RIG
            f = [0.6 * W if i != 1 else 1.4 * W for i in range(len(rig))]
            dist = np.zeros((len(rig), 5), np.float32)
        else:
            rig = FISHEYE_RIG
            f = [W / math.pi * 1.05] * len(rig)
            dist = np.tile(np.array([[0.05, -0.01, 0.002, -0.0005]], np.float32), (len(rig), 1))
        intr = np.array([[fi, fi, W / 2, H / 2] for fi in f], np.float32)
        T = np.stack([cam2ego_matrix(*r) for r in rig])
        return dict(intrinsics=torch.from_numpy(intr), distortion=torch.from_numpy(dist), cam2ego=torch.from_numpy(T))

    def _frame(self, g: torch.Generator) -> Dict:
        out = {}
        for group, (H, W) in self.img_sizes.items():
            calib = self._calib(group, H, W)
            N = calib["intrinsics"].shape[0]
            out[f"img_{group}"] = torch.randn(N, 3, H, W, generator=g)
            out[f"calib_{group}"] = calib
        lo = torch.tensor(self.grid.lower)
        hi = torch.tensor(self.grid.upper)
        if self.with_lidar:
            n = 20000
            xyz = lo + torch.rand(n, 3, generator=g) * (hi - lo)
            out["lidar_points"] = torch.cat([xyz, torch.rand(n, 1, generator=g)], 1)
        if self.with_radar:
            n = 300
            xyz = lo + torch.rand(n, 3, generator=g) * (hi - lo)
            out["radar_points"] = torch.cat([xyz, torch.randn(n, 2, generator=g), torch.rand(n, 2, generator=g)], 1)
        return out

    def __getitem__(self, idx):
        g = torch.Generator().manual_seed(self.seed * 100003 + idx)
        s = self._frame(g)
        lo = torch.tensor(self.grid.lower)
        hi = torch.tensor(self.grid.upper)

        ang = torch.rand(1, generator=g).item() * 2 * math.pi
        slot_center = torch.tensor([math.cos(ang), math.sin(ang)]) * 5.0
        half = torch.tensor([[-1.25, 2.6], [1.25, 2.6], [1.25, -2.6], [-1.25, -2.6]]) / 2 * torch.tensor([2.0, 1.0])
        slot = slot_center + half
        s["nav"] = dict(
            target_slot=slot.float(), target_valid=torch.tensor(True),
            route=torch.randn(self.P, 2, generator=g) * 3, route_valid=torch.rand(self.P, generator=g) > 0.2,
            ego_status=torch.randn(self.E, generator=g), phase=torch.tensor(1),
        )

        for group, (H, W) in self.img_sizes.items():
            N = s[f"img_{group}"].shape[0]
            depth = torch.rand(N, H, W, generator=g) * 15 + 0.5
            depth[torch.rand(N, H, W, generator=g) > 0.05] = 0
            s[f"gt_depth_{group}"] = depth
            s[f"gt_pvseg_{group}"] = torch.randint(0, self.k_pvseg, (N, H, W), generator=g)
            dets = []
            for _ in range(N):
                M = 6
                d = torch.zeros(M, 7)
                d[:, 0] = torch.randint(0, self.k_pvdet, (M,), generator=g).float()
                d[:, 1] = torch.rand(M, generator=g) * W
                d[:, 2] = torch.rand(M, generator=g) * H
                d[:, 3:5] = (d[:, :1] != 0).float().expand(M, 2) * 40.0
                a = torch.rand(M, generator=g) * 2 * math.pi
                d[:, 5], d[:, 6] = a.cos(), a.sin()
                dets.append(d)
            s[f"gt_pvdet_{group}"] = dets

        Hs, Ws = self.grid.ny * self.up, self.grid.nx * self.up
        s["gt_bev_seg"] = torch.randint(0, self.k_bev, (Hs, Ws), generator=g)
        hgt = torch.rand(2, Hs, Ws, generator=g).sort(0)[0] * 2.0
        s["gt_bev_height"] = hgt

        M = 10
        boxes = torch.zeros(M, 9)
        boxes[:, :2] = lo[:2] + torch.rand(M, 2, generator=g) * (hi[:2] - lo[:2])
        boxes[:, 2] = 0.8
        boxes[:, 3:6] = torch.tensor([4.6, 1.9, 1.6])
        boxes[:, 6] = torch.rand(M, generator=g) * 2 * math.pi - math.pi
        boxes[:, 7:9] = torch.randn(M, 2, generator=g)
        s["gt_boxes_3d"] = boxes
        s["gt_labels_3d"] = torch.randint(0, self.k_det, (M,), generator=g)
        s["gt_attr_3d"] = torch.where(s["gt_labels_3d"] == 3, torch.randint(0, 2, (M,), generator=g), -torch.ones(M, dtype=torch.long))

        def rand_pts(n):
            return lo + torch.rand(n, 3, generator=g) * (hi - lo)

        s["occ_pts"] = rand_pts(self.n_occ)
        lab = torch.randint(0, 2, (self.n_occ,), generator=g)
        lab[torch.rand(self.n_occ, generator=g) < 0.05] = 255
        s["occ_label"] = lab
        s["occ_sem"] = torch.where(lab == 1, torch.randint(0, self.k_occ_sem, (self.n_occ,), generator=g),
                                   torch.full((self.n_occ,), 255, dtype=torch.long))
        s["flow_pts"] = rand_pts(self.n_flow)
        s["flow_val"] = torch.randn(self.n_flow, 3, generator=g) * 0.5
        s["flow_mask"] = torch.rand(self.n_flow, generator=g) > 0.1
        s["sdf_pts"] = rand_pts(self.n_sdf)
        s["sdf_val"] = (torch.rand(self.n_sdf, generator=g) - 0.5) * 1.2
        s["sdf_mask"] = torch.rand(self.n_sdf, generator=g) > 0.1

        s["gt_slots"] = torch.stack([slot, slot + torch.tensor([2.6, 0.0])]).float()
        s["gt_slot_occupied"] = torch.tensor([0.0, 1.0])

        t = torch.linspace(0, 1, self.T)
        x = slot_center[0] * t
        y = slot_center[1] * t
        yaw = torch.full_like(t, ang)
        s["gt_traj"] = torch.stack([x, y, yaw.cos(), yaw.sin()], -1)
        gear = torch.zeros(self.T, dtype=torch.long)
        gear[self.T // 2:] = 1
        gear[-2:] = 2
        s["gt_gear"] = gear
        s["gt_traj_mask"] = torch.ones(self.T, dtype=torch.bool)

        if self.num_history > 0:
            s["history"] = [self._frame(g) for _ in range(self.num_history)]
            Ts = torch.eye(4).repeat(self.num_history, 1, 1)
            for k in range(self.num_history):
                Ts[k, 0, 3] = -0.3 * (k + 1)
            s["T_cur_from_hist"] = Ts
            s["hist_valid"] = torch.ones(self.num_history, dtype=torch.bool)
        return s
