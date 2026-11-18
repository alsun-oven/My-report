from typing import Any, Dict, List

import numpy as np
import torch

# 每个样本长度不同，保持 list，不 stack
LIST_KEYS = {
    "lidar_points", "radar_points",
    "gt_boxes_3d", "gt_labels_3d", "gt_attr_3d",
    "gt_slots", "gt_slot_occupied",
    "meta",
}


def collate_parking(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k in samples[0]:
        vals = [s[k] for s in samples]
        if k == "history":
            T = len(vals[0])
            assert all(len(v) == T for v in vals), "all samples must carry the same number of history frames"
            out[k] = [collate_parking([v[t] for v in vals]) for t in range(T)]
        elif k in LIST_KEYS or k.startswith("gt_pvdet_"):
            out[k] = vals
        elif torch.is_tensor(vals[0]):
            out[k] = torch.stack(vals)
        elif isinstance(vals[0], np.ndarray):
            out[k] = torch.stack([torch.from_numpy(v) for v in vals])
        elif isinstance(vals[0], dict):
            out[k] = collate_parking(vals)
        elif isinstance(vals[0], (bool, int, float)):
            out[k] = torch.tensor(vals)
        else:
            out[k] = vals
    return out


def move_to_device(obj: Any, device) -> Any:
    if torch.is_tensor(obj):
        return obj.to(device, non_blocking=True)
    if isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, list):
        return [move_to_device(v, device) for v in obj]
    return obj
