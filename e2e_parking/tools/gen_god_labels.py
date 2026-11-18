"""为一帧生成 GOD 标签（occ / sem / sdf / flow，两级分辨率）。

输入 npz：
  points  (N, 3+)  多帧累积激光点，自车系，动态物体已按框速度挪到当前时刻
  origins (N, 3)   每个点对应的传感器原点（自车系）
  boxes   (M, 9)   当前帧框 x, y, z, l, w, h, yaw, vx, vy（可选）
  labels  (M,)     框类别（可选）
用法：python tools/gen_god_labels.py in.npz out.npz
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from parking_e2e.data.god_labels import build_god_labels  # noqa: E402

LEVELS = dict(
    near=np.array([[-5.0, 5.0, 0.05], [-5.0, 5.0, 0.05], [-0.4, 2.2, 0.05]], np.float32),
    far=np.array([[-12.0, 12.0, 0.10], [-12.0, 12.0, 0.10], [-0.4, 2.2, 0.10]], np.float32),
)
# 检测类别 -> occ 细分类（vehicle, pillar, curb, other）；没映射的占据都算 other
LABEL_TO_SEM = {0: 0, 2: 0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp")
    ap.add_argument("out")
    ap.add_argument("--tau", type=float, default=0.4)
    args = ap.parse_args()
    d = np.load(args.inp)
    boxes = d["boxes"] if "boxes" in d.files else None
    box_labels = d["labels"] if "labels" in d.files else None
    labels = build_god_labels(d["points"], d["origins"], LEVELS, boxes, box_labels,
                              label_to_sem=LABEL_TO_SEM, other_sem=3, dynamic_labels=(0, 1, 2), tau=args.tau)
    np.savez_compressed(args.out, **labels)
    for k, v in labels.items():
        print(k, v.shape, v.dtype)


if __name__ == "__main__":
    main()
