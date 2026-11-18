import pickle
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from ..registry import DATASETS, TRANSFORMS


class Compose:
    def __init__(self, transforms: Sequence[dict]):
        self.transforms = [TRANSFORMS.build(t) for t in transforms]

    def __call__(self, results):
        for t in self.transforms:
            results = t(results)
        return results


@DATASETS.register_module()
class ParkingE2EDataset(Dataset):
    """ann_file 是 pickle 的 list[dict]，每帧一条：

    token, scene_token, timestamp
    prev_token                 上一帧（同一 scene），没有写 None
    ego2global (4, 4)
    cams: {cam_name: dict(img_path, intrinsics [fx, fy, cx, cy], distortion [...], cam2ego (4, 4))}
          针孔 distortion 为 OpenCV (k1, k2, p1, p2, k3)，鱼眼为 KB (k1, k2, k3, k4)
    lidar_path, radar_path     自车系点云（.npy / .bin float32）
    nav: dict(target_slot (4, 2) | None, route (P, 2), route_valid (P,), ego_status (E,), phase)
    ann: dict(
        boxes_3d (M, 9), labels_3d (M,), attr_3d (M,)
        slots (S, 4, 2), slot_occupied (S,)
        traj (T, 4), gear (T,), traj_mask (T,)
        bev_seg_path, bev_height_path, god_path
        pvseg_paths {cam_name: path}, pvdet {cam_name: (M, 7)}
    )
    """

    def __init__(self, ann_file: str, pipeline: Sequence[dict], history_pipeline: Optional[Sequence[dict]] = None,
                 num_history: int = 0, history_stride: int = 1, test_mode: bool = False):
        with open(ann_file, "rb") as f:
            self.infos: List[Dict] = pickle.load(f)
        self.token_to_idx = {info["token"]: i for i, info in enumerate(self.infos)}
        self.pipeline = Compose(pipeline)
        self.history_pipeline = Compose(history_pipeline) if history_pipeline else None
        self.num_history = num_history
        self.history_stride = history_stride
        self.test_mode = test_mode

    def __len__(self):
        return len(self.infos)

    def _history_indices(self, idx: int) -> List[Optional[int]]:
        out, cur = [], idx
        for _ in range(self.num_history):
            for _ in range(self.history_stride):
                prev = self.infos[cur].get("prev_token") if cur is not None else None
                cur = self.token_to_idx.get(prev) if prev is not None else None
            out.append(cur)
        return out

    def __getitem__(self, idx):
        info = self.infos[idx]
        sample = self.pipeline(dict(info=info))
        if self.num_history > 0:
            assert self.history_pipeline is not None, "num_history > 0 needs history_pipeline"
            cur_pose = np.asarray(info["ego2global"], np.float64)
            hist, Ts, valid = [], [], []
            for h in self._history_indices(idx):
                ok = h is not None
                hinfo = self.infos[h] if ok else info
                hist.append(self.history_pipeline(dict(info=hinfo)))
                T = np.linalg.inv(cur_pose) @ np.asarray(hinfo["ego2global"], np.float64)
                Ts.append(T.astype(np.float32))
                valid.append(ok)
            sample["history"] = hist
            sample["T_cur_from_hist"] = torch.from_numpy(np.stack(Ts))
            sample["hist_valid"] = torch.tensor(valid)
        return sample
