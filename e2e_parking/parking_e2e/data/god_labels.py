"""GOD 标签离线生成：射线投射的占据（含未知）、截断 SDF、由框速度得到的 occ flow 与细分类。

输入是多帧累积、已经把动态物体按框速度挪到当前时刻的激光点（自车系），以及每个点对应的传感器原点。
"""
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

FREE, OCC, UNKNOWN = 0, 1, 255


def grid_shape(bounds: np.ndarray) -> Tuple[int, int, int]:
    """bounds (3, 3) 每行 (min, max, res)，返回 (Z, Y, X)。"""
    n = np.round((bounds[:, 1] - bounds[:, 0]) / bounds[:, 2]).astype(int)
    return int(n[2]), int(n[1]), int(n[0])


def _to_index(pts: np.ndarray, bounds: np.ndarray):
    idx = np.floor((pts - bounds[:, 0]) / bounds[:, 2]).astype(np.int64)
    Z, Y, X = grid_shape(bounds)
    valid = (idx[:, 0] >= 0) & (idx[:, 0] < X) & (idx[:, 1] >= 0) & (idx[:, 1] < Y) & (idx[:, 2] >= 0) & (idx[:, 2] < Z)
    return idx, valid


def occupancy_by_raycast(points: np.ndarray, origins: np.ndarray, bounds: np.ndarray,
                         step_ratio: float = 0.5, chunk: int = 20000) -> np.ndarray:
    """射线从原点走到命中点之前一格，经过的体素标空；命中点体素标占据；其余保持未知。

    自车底下、立柱背面、射线终点之外都保持未知，训练时不进损失。
    """
    Z, Y, X = grid_shape(bounds)
    occ = np.full((Z, Y, X), UNKNOWN, np.uint8)
    free = np.zeros((Z, Y, X), bool)
    step = float(bounds[:, 2].min()) * step_ratio
    stop = float(bounds[:, 2].max())
    corner = float(np.linalg.norm(np.abs(bounds[:, :2]).max(1)))
    for s in range(0, len(points), chunk):
        p, o = points[s:s + chunk, :3].astype(np.float32), origins[s:s + chunk, :3].astype(np.float32)
        vec = p - o
        dist = np.linalg.norm(vec, axis=1)
        d = vec / np.maximum(dist, 1e-6)[:, None]
        # 射线离开网格后不再步进，远处的命中点不会把步数撑大
        t_end = np.minimum(dist - stop, corner + np.linalg.norm(o, axis=1))
        n_steps = int(np.ceil(max(t_end.max(), 0) / step)) + 1
        t = np.arange(n_steps, dtype=np.float32) * step
        ok = t[None, :] < t_end[:, None]
        samples = o[:, None, :] + d[:, None, :] * t[None, :, None]
        samples = samples[ok]
        idx, valid = _to_index(samples, bounds)
        idx = idx[valid]
        free[idx[:, 2], idx[:, 1], idx[:, 0]] = True
    occ[free] = FREE
    idx, valid = _to_index(points[:, :3], bounds)
    idx = idx[valid]
    occ[idx[:, 2], idx[:, 1], idx[:, 0]] = OCC
    return occ


def tsdf_from_occupancy(occ: np.ndarray, bounds: np.ndarray, tau: float = 0.4) -> np.ndarray:
    """有符号距离（米），内部为负。未知体素、以及离未知区比离表面更近的空体素记 NaN：
    那里的距离可能被未看见的障碍改写，不能当真值。
    """
    from scipy.ndimage import distance_transform_edt

    res_zyx = (bounds[2, 2], bounds[1, 2], bounds[0, 2])
    half = 0.5 * min(res_zyx)
    is_occ = occ == OCC
    is_unk = occ == UNKNOWN
    if not is_occ.any():
        sdf = np.full(occ.shape, tau, np.float32)
    else:
        d_out = distance_transform_edt(~is_occ, sampling=res_zyx)
        d_in = distance_transform_edt(is_occ, sampling=res_zyx)
        sdf = np.where(is_occ, -(d_in - half), d_out - half).astype(np.float32)
    if is_unk.any():
        d_unk = distance_transform_edt(~is_unk, sampling=res_zyx)
        untrusted = (~is_occ) & (d_unk < np.minimum(np.abs(sdf), tau))
        sdf[untrusted] = np.nan
    sdf[is_unk] = np.nan
    return sdf


def _points_in_box(pts: np.ndarray, box: np.ndarray) -> np.ndarray:
    x, y, z, l, w, h, yaw = box[:7]
    c, s = np.cos(yaw), np.sin(yaw)
    dx, dy = pts[:, 0] - x, pts[:, 1] - y
    lx = c * dx + s * dy
    ly = -s * dx + c * dy
    return (np.abs(lx) <= l / 2) & (np.abs(ly) <= w / 2) & (np.abs(pts[:, 2] - z) <= h / 2)


def semantics_and_flow(occ: np.ndarray, bounds: np.ndarray, boxes: np.ndarray, labels: np.ndarray,
                       label_to_sem: Dict[int, int], other_sem: int, dynamic_labels: Sequence[int],
                       box_margin: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    """占据体素落在框内则取框类别映射的细分类，动态类的流取框速度；其余占据为 other_sem、流为 0。"""
    sem = np.full(occ.shape, UNKNOWN, np.uint8)
    flow = np.zeros((3,) + occ.shape, np.float16)
    zi, yi, xi = np.nonzero(occ == OCC)
    sem[zi, yi, xi] = other_sem
    if len(zi) == 0 or len(boxes) == 0:
        return sem, flow
    centers = np.stack([bounds[0, 0] + (xi + 0.5) * bounds[0, 2], bounds[1, 0] + (yi + 0.5) * bounds[1, 2],
                        bounds[2, 0] + (zi + 0.5) * bounds[2, 2]], 1)
    dyn = set(int(d) for d in dynamic_labels)
    for box, lab in zip(boxes, labels):
        b = box.copy()
        b[3:6] += 2 * box_margin
        m = _points_in_box(centers, b)
        if not m.any():
            continue
        if int(lab) in label_to_sem:
            sem[zi[m], yi[m], xi[m]] = label_to_sem[int(lab)]
        if int(lab) in dyn and len(box) >= 9:
            flow[0, zi[m], yi[m], xi[m]] = box[7]
            flow[1, zi[m], yi[m], xi[m]] = box[8]
    return sem, flow


def build_god_labels(points: np.ndarray, origins: np.ndarray, levels: Dict[str, np.ndarray],
                     boxes: Optional[np.ndarray] = None, labels: Optional[np.ndarray] = None,
                     label_to_sem: Optional[Dict[int, int]] = None, other_sem: int = 3,
                     dynamic_labels: Sequence[int] = (0, 1, 2), tau: float = 0.4) -> Dict[str, np.ndarray]:
    """levels: {"near": bounds, "far": bounds}。返回可直接 np.savez_compressed 的字典。"""
    out = {}
    boxes = np.zeros((0, 9), np.float32) if boxes is None else boxes
    labels = np.zeros(0, np.int64) if labels is None else labels
    for name, b in levels.items():
        b = np.asarray(b, np.float32)
        occ = occupancy_by_raycast(points, origins, b)
        out[f"{name}_bounds"] = b
        out[f"{name}_occ"] = occ
        out[f"{name}_sdf"] = tsdf_from_occupancy(occ, b, tau).astype(np.float16)
        sem, flow = semantics_and_flow(occ, b, boxes, labels, label_to_sem or {}, other_sem, dynamic_labels)
        out[f"{name}_sem"] = sem
        out[f"{name}_flow"] = flow
    return out
