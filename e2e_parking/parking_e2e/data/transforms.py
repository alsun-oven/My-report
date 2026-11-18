"""真实数据的 pipeline。每个 transform 接收并返回 results 字典（numpy），最后 FormatBundle 转成模型输入。

results 在 LoadMultiViewImages 之后的约定：
  imgs[group]     list[N] of (H, W, 3) uint8 RGB
  calib[group]    dict(intrinsics (N, 4), distortion (N, K), cam2ego (N, 4, 4))，intrinsics 随缩放裁剪同步更新
  cam_names[group]
  pvseg[group]    list[N] of (H, W) uint8（可选）
  pvdet[group]    list[N] of (M, 7)（可选）
"""
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from ..registry import CAMERA_MODELS, TRANSFORMS


def _imread(path: str) -> np.ndarray:
    try:
        import cv2
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(path)
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    except ImportError:
        from PIL import Image
        return np.array(Image.open(path))


def _resize(img: np.ndarray, size_wh: Tuple[int, int], nearest: bool = False) -> np.ndarray:
    try:
        import cv2
        return cv2.resize(img, size_wh, interpolation=cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR)
    except ImportError:
        from PIL import Image
        return np.array(Image.fromarray(img).resize(size_wh, Image.NEAREST if nearest else Image.BILINEAR))


def _load_array(path_or_array, data_root: str = "") -> np.ndarray:
    if isinstance(path_or_array, str):
        p = os.path.join(data_root, path_or_array)
        if p.endswith(".npy"):
            return np.load(p)
        if p.endswith(".bin"):
            return np.fromfile(p, dtype=np.float32)
        return _imread(p)
    return np.asarray(path_or_array)


@TRANSFORMS.register_module()
class LoadMultiViewImages:
    def __init__(self, cam_order: Dict[str, Sequence[str]], data_root: str = ""):
        self.cam_order = {g: list(v) for g, v in cam_order.items()}
        self.data_root = data_root

    def __call__(self, results):
        info = results["info"]
        results["imgs"], results["calib"], results["cam_names"] = {}, {}, {}
        for g, names in self.cam_order.items():
            imgs, intr, dist, ext = [], [], [], []
            for n in names:
                c = info["cams"][n]
                imgs.append(_imread(os.path.join(self.data_root, c["img_path"])))
                intr.append(np.asarray(c["intrinsics"], np.float32))
                dist.append(np.asarray(c.get("distortion", []), np.float32))
                ext.append(np.asarray(c["cam2ego"], np.float32))
            results["imgs"][g] = imgs
            results["calib"][g] = dict(intrinsics=np.stack(intr), distortion=np.stack(dist), cam2ego=np.stack(ext))
            results["cam_names"][g] = names
        return results


@TRANSFORMS.register_module()
class LoadPoints:
    """点云已在自车系。key='lidar' 读 info['lidar_path']，key='radar' 读 info['radar_path']。"""

    def __init__(self, key: str = "lidar", dims: int = 4, data_root: str = ""):
        self.key, self.dims, self.data_root = key, dims, data_root

    def __call__(self, results):
        path = results["info"].get(f"{self.key}_path")
        if path is None:
            return results
        pts = _load_array(path, self.data_root).astype(np.float32)
        results[f"{self.key}_points"] = pts.reshape(-1, self.dims) if pts.ndim == 1 else pts[:, :self.dims]
        return results


@TRANSFORMS.register_module()
class LoadAnnotations:
    def __init__(self, data_root: str = "", with_pv: bool = True, with_bev: bool = True,
                 with_god: bool = True, with_plan: bool = True):
        self.data_root = data_root
        self.with_pv, self.with_bev, self.with_god, self.with_plan = with_pv, with_bev, with_god, with_plan

    def __call__(self, results):
        ann = results["info"]["ann"]
        if self.with_pv:
            results["pvseg"], results["pvdet"] = {}, {}
            for g, names in results["cam_names"].items():
                segs = [ann.get("pvseg_paths", {}).get(n) for n in names]
                if all(s is not None for s in segs):
                    results["pvseg"][g] = [_load_array(s, self.data_root).astype(np.uint8) for s in segs]
                results["pvdet"][g] = [np.asarray(ann.get("pvdet", {}).get(n, np.zeros((0, 7))), np.float32).reshape(-1, 7)
                                       for n in names]
        if self.with_bev:
            results["gt_boxes_3d"] = np.asarray(ann["boxes_3d"], np.float32).reshape(-1, 9)
            results["gt_labels_3d"] = np.asarray(ann["labels_3d"], np.int64).reshape(-1)
            results["gt_attr_3d"] = np.asarray(ann.get("attr_3d", -np.ones(len(results["gt_labels_3d"]))), np.int64).reshape(-1)
            if "bev_seg_path" in ann:
                results["gt_bev_seg"] = _load_array(ann["bev_seg_path"], self.data_root).astype(np.int64)
            if "bev_height_path" in ann:
                results["gt_bev_height"] = _load_array(ann["bev_height_path"], self.data_root).astype(np.float32)
        if self.with_god and "god_path" in ann:
            results["god"] = dict(np.load(os.path.join(self.data_root, ann["god_path"])))
        if self.with_plan:
            results["gt_slots"] = np.asarray(ann.get("slots", np.zeros((0, 4, 2))), np.float32).reshape(-1, 4, 2)
            results["gt_slot_occupied"] = np.asarray(ann.get("slot_occupied", np.zeros(0)), np.float32).reshape(-1)
            results["gt_traj"] = np.asarray(ann["traj"], np.float32)
            results["gt_gear"] = np.asarray(ann["gear"], np.int64)
            results["gt_traj_mask"] = np.asarray(ann.get("traj_mask", np.ones(len(ann["gear"]))), bool)
        return results


@TRANSFORMS.register_module()
class ResizeCropMultiView:
    """按组缩放到目标尺寸，可选训练时尺度抖动 + 随机裁剪。intrinsics 同步更新：
    缩放乘到 fx, fy, cx, cy，裁剪平移 cx, cy。KB 鱼眼的畸变作用在归一化角度上，这两个操作对它同样成立。
    """

    def __init__(self, sizes: Dict[str, Sequence[int]], scale_jitter: Tuple[float, float] = (1.0, 1.0),
                 train: bool = False):
        self.sizes = {g: tuple(v) for g, v in sizes.items()}
        self.scale_jitter = scale_jitter
        self.train = train

    def __call__(self, results):
        for g, (H, W) in self.sizes.items():
            if g not in results["imgs"]:
                continue
            calib = results["calib"][g]
            intr = calib["intrinsics"].copy()
            for i, img in enumerate(results["imgs"][g]):
                h0, w0 = img.shape[:2]
                s = max(H / h0, W / w0)
                if self.train:
                    s *= np.random.uniform(*self.scale_jitter)
                nh, nw = int(round(h0 * s)), int(round(w0 * s))
                nh, nw = max(nh, H), max(nw, W)
                sx, sy = nw / w0, nh / h0
                ox = np.random.randint(0, nw - W + 1) if self.train else (nw - W) // 2
                oy = np.random.randint(0, nh - H + 1) if self.train else (nh - H) // 2
                results["imgs"][g][i] = _resize(img, (nw, nh))[oy:oy + H, ox:ox + W]
                intr[i] = [intr[i, 0] * sx, intr[i, 1] * sy, intr[i, 2] * sx - ox, intr[i, 3] * sy - oy]
                if g in results.get("pvseg", {}):
                    results["pvseg"][g][i] = _resize(results["pvseg"][g][i], (nw, nh), nearest=True)[oy:oy + H, ox:ox + W]
                if g in results.get("pvdet", {}) and len(results["pvdet"][g][i]):
                    d = results["pvdet"][g][i].copy()
                    d[:, 1] = d[:, 1] * sx - ox
                    d[:, 2] = d[:, 2] * sy - oy
                    d[:, 3] *= sx
                    d[:, 4] *= sy
                    keep = (d[:, 1] >= 0) & (d[:, 1] < W) & (d[:, 2] >= 0) & (d[:, 2] < H)
                    results["pvdet"][g][i] = d[keep]
            calib["intrinsics"] = intr.astype(np.float32)
        return results


@TRANSFORMS.register_module()
class DepthFromLidar:
    """激光点投影到每路相机生成稀疏深度真值，z-buffer 取最近。针孔写 z-depth，鱼眼写 range。
    放在 ResizeCropMultiView 之后，用更新过的 intrinsics。ego_mask_paths 可以把自车车身区域置 0。
    """

    def __init__(self, camera_models: Dict[str, dict], min_depth: float = 0.2, max_depth: float = 30.0):
        self.cams = {g: CAMERA_MODELS.build(c) for g, c in camera_models.items()}
        self.min_depth, self.max_depth = min_depth, max_depth

    def __call__(self, results):
        pts = results.get("lidar_points")
        if pts is None:
            return results
        xyz = torch.from_numpy(pts[:, :3]).float()
        results["gt_depth"] = {}
        for g, imgs in results["imgs"].items():
            calib = results["calib"][g]
            maps = []
            for i, img in enumerate(imgs):
                H, W = img.shape[:2]
                ego2cam = torch.from_numpy(np.linalg.inv(calib["cam2ego"][i])).float()
                pc = xyz @ ego2cam[:3, :3].T + ego2cam[:3, 3]
                intr = torch.from_numpy(calib["intrinsics"][i]).float().expand(len(pc), 4)
                dist = calib["distortion"][i]
                dist = torch.from_numpy(dist).float().expand(len(pc), len(dist)) if len(dist) else None
                uv, depth, valid = self.cams[g].project(pc, intr, dist)
                u = uv[:, 0].round().long()
                v = uv[:, 1].round().long()
                valid &= (u >= 0) & (u < W) & (v >= 0) & (v < H) & (depth > self.min_depth) & (depth < self.max_depth)
                u, v, depth = u[valid], v[valid], depth[valid]
                dmap = torch.full((H * W,), float("inf"))
                dmap.scatter_reduce_(0, v * W + u, depth, reduce="amin")
                dmap[torch.isinf(dmap)] = 0
                maps.append(dmap.view(H, W).numpy())
            results["gt_depth"][g] = maps
        return results


@TRANSFORMS.register_module()
class SampleGODQueries:
    """从多分辨率 GOD 标签网格采样查询点。未知体素（255 / NaN）不采样。

    god npz 键（tools/gen_god_labels.py 生成）：
      {level}_bounds (3, 3) 每行 (min, max, res)，level ∈ levels（默认 near 5 cm ±5 m、far 10 cm ±12 m）
      {level}_occ (Z, Y, X) uint8 0 空 / 1 占据 / 255 未知
      {level}_sem (Z, Y, X) uint8 占据细分类，255 未知
      {level}_sdf (Z, Y, X) float16 米，NaN 无效
      {level}_flow (3, Z, Y, X) float16 m/s
    far 层落在 near 范围内的体素不采样，避免重复计数。
    """

    def __init__(self, num_occ: int = 16384, num_sdf: int = 16384, num_flow: int = 4096, pos_ratio: float = 0.5,
                 near_ratio: float = 0.6, tau: float = 0.4, sdf_band_ratio: float = 0.8,
                 levels: Sequence[str] = ("near", "far"), jitter: bool = True):
        self.num_occ, self.num_sdf, self.num_flow = num_occ, num_sdf, num_flow
        self.pos_ratio, self.near_ratio, self.tau, self.band_ratio = pos_ratio, near_ratio, tau, sdf_band_ratio
        self.levels = list(levels)
        self.jitter = jitter

    @staticmethod
    def _centers(idx, bounds):
        z, y, x = idx
        b = bounds
        return np.stack([b[0, 0] + (x + 0.5) * b[0, 2], b[1, 0] + (y + 0.5) * b[1, 2], b[2, 0] + (z + 0.5) * b[2, 2]], -1)

    @staticmethod
    def _choose(n_avail, n):
        if n_avail == 0 or n <= 0:
            return np.zeros(0, np.int64)
        return np.random.choice(n_avail, n, replace=n_avail < n)

    def _level_quota(self, total):
        if len(self.levels) == 1:
            return {self.levels[0]: total}
        near = int(total * self.near_ratio)
        rest = (total - near) // (len(self.levels) - 1)
        return {l: (near if i == 0 else rest) for i, l in enumerate(self.levels)}

    def _outside_mask(self, god, level, shape):
        if level == self.levels[0]:
            return np.ones(shape, bool)
        nb = god[f"{self.levels[0]}_bounds"]
        b = god[f"{level}_bounds"]
        zz, yy, xx = np.meshgrid(*[np.arange(s) for s in shape], indexing="ij")
        c = self._centers((zz, yy, xx), b)
        inside = np.ones(shape, bool)
        for k in range(3):
            inside &= (c[..., k] >= nb[k, 0]) & (c[..., k] < nb[k, 1])
        return ~inside

    def __call__(self, results):
        god = results.pop("god", None)
        if god is None:
            return results
        occ_p, occ_l, occ_s, sdf_p, sdf_v, flow_p, flow_v = [], [], [], [], [], [], []
        q_occ, q_sdf, q_flow = self._level_quota(self.num_occ), self._level_quota(self.num_sdf), self._level_quota(self.num_flow)
        for level in self.levels:
            b = god[f"{level}_bounds"].astype(np.float32)
            occ = god[f"{level}_occ"]
            allowed = self._outside_mask(god, level, occ.shape)
            res = b[:, 2]

            def pts_of(idx, jitter):
                c = self._centers(idx, b)
                if jitter and self.jitter:
                    c = c + (np.random.rand(*c.shape) - 0.5) * res
                return c.astype(np.float32)

            pos = np.nonzero((occ == 1) & allowed)
            neg = np.nonzero((occ == 0) & allowed)
            n_pos = int(q_occ[level] * self.pos_ratio)
            ip, ineg = self._choose(len(pos[0]), n_pos), self._choose(len(neg[0]), q_occ[level] - n_pos)
            sel_p = tuple(a[ip] for a in pos)
            sel_n = tuple(a[ineg] for a in neg)
            occ_p += [pts_of(sel_p, True), pts_of(sel_n, True)]
            occ_l += [np.ones(len(ip), np.int64), np.zeros(len(ineg), np.int64)]
            sem = god.get(f"{level}_sem")
            occ_s += [sem[sel_p].astype(np.int64) if sem is not None else np.full(len(ip), 255, np.int64),
                      np.full(len(ineg), 255, np.int64)]

            sdf = god.get(f"{level}_sdf")
            if sdf is not None:
                sdf = sdf.astype(np.float32)
                ok = np.isfinite(sdf) & allowed
                band = np.nonzero(ok & (np.abs(sdf) < self.tau))
                out = np.nonzero(ok & (np.abs(sdf) >= self.tau))
                nb = int(q_sdf[level] * self.band_ratio)
                ib, io = self._choose(len(band[0]), nb), self._choose(len(out[0]), q_sdf[level] - nb)
                for src, ii in ((band, ib), (out, io)):
                    sel = tuple(a[ii] for a in src)
                    sdf_p.append(pts_of(sel, False))
                    sdf_v.append(sdf[sel])

            flow = god.get(f"{level}_flow")
            if flow is not None:
                cand = np.nonzero((occ == 1) & allowed)
                ii = self._choose(len(cand[0]), q_flow[level])
                sel = tuple(a[ii] for a in cand)
                flow_p.append(pts_of(sel, False))
                flow_v.append(flow.astype(np.float32)[(slice(None),) + sel].T)

        def pad(arrs, n, fill, tail=()):
            a = np.concatenate(arrs, 0) if arrs else np.zeros((0,) + tail, np.float32)
            m = np.zeros(n, bool)
            m[:min(len(a), n)] = True
            out = np.full((n,) + a.shape[1:], fill, dtype=a.dtype)
            out[:min(len(a), n)] = a[:n]
            return out, m

        results["occ_pts"], _ = pad(occ_p, self.num_occ, 0.0, (3,))
        results["occ_label"], _ = pad(occ_l, self.num_occ, 255)
        results["occ_sem"], _ = pad(occ_s, self.num_occ, 255)
        results["sdf_pts"], results["sdf_mask"] = pad(sdf_p, self.num_sdf, 0.0, (3,))
        results["sdf_val"], _ = pad(sdf_v, self.num_sdf, 0.0)
        results["flow_pts"], results["flow_mask"] = pad(flow_p, self.num_flow, 0.0, (3,))
        results["flow_val"], _ = pad(flow_v, self.num_flow, 0.0, (3,))
        return results


@TRANSFORMS.register_module()
class FormatBundle:
    """numpy -> 模型输入。图像归一化后排成 (N, 3, H, W)。"""

    def __init__(self, mean=(123.675, 116.28, 103.53), std=(58.395, 57.12, 57.375), history: bool = False):
        self.mean = np.asarray(mean, np.float32)
        self.std = np.asarray(std, np.float32)
        self.history = history

    def __call__(self, results):
        out = {}
        for g, imgs in results["imgs"].items():
            arr = (np.stack(imgs).astype(np.float32) - self.mean) / self.std
            out[f"img_{g}"] = torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous()
            out[f"calib_{g}"] = {k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in results["calib"][g].items()}
        for k in ("lidar_points", "radar_points"):
            if k in results:
                out[k] = torch.from_numpy(results[k])
        if self.history:
            return out

        info = results["info"]
        nav = info["nav"]
        slot = nav.get("target_slot")
        route = np.asarray(nav["route"], np.float32).reshape(-1, 2)
        out["nav"] = dict(
            target_slot=torch.from_numpy(np.asarray(slot if slot is not None else np.zeros((4, 2)), np.float32)),
            target_valid=torch.tensor(slot is not None),
            route=torch.from_numpy(route),
            route_valid=torch.from_numpy(np.asarray(nav.get("route_valid", np.ones(len(route))), bool)),
            ego_status=torch.from_numpy(np.asarray(nav["ego_status"], np.float32)),
            phase=torch.tensor(int(nav.get("phase", 1))),
        )
        for g, maps in results.get("gt_depth", {}).items():
            out[f"gt_depth_{g}"] = torch.from_numpy(np.stack(maps).astype(np.float32))
        for g, masks in results.get("pvseg", {}).items():
            out[f"gt_pvseg_{g}"] = torch.from_numpy(np.stack(masks).astype(np.int64))
        for g, dets in results.get("pvdet", {}).items():
            out[f"gt_pvdet_{g}"] = [torch.from_numpy(d) for d in dets]
        tensor_keys = ["gt_boxes_3d", "gt_labels_3d", "gt_attr_3d", "gt_bev_seg", "gt_bev_height",
                       "occ_pts", "occ_label", "occ_sem", "sdf_pts", "sdf_val", "sdf_mask",
                       "flow_pts", "flow_val", "flow_mask",
                       "gt_slots", "gt_slot_occupied", "gt_traj", "gt_gear", "gt_traj_mask"]
        for k in tensor_keys:
            if k in results:
                out[k] = torch.from_numpy(np.ascontiguousarray(results[k]))
        out["meta"] = dict(token=info.get("token"), cam_names=results["cam_names"])
        return out
