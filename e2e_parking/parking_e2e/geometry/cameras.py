"""相机模型。所有张量最后一维是坐标，intrinsics 为 (fx, fy, cx, cy)，调用方负责把它们广播到同一前缀形状。

深度定义随模型不同：
- Pinhole：depth 是相机系 z（光轴方向）。unproject 返回 z=1 的射线，点 = ray * depth。
- KannalaBrandt 鱼眼：depth 是到光心的距离（range）。FOV 接近或超过 180° 时 z 会 ≤ 0，z-depth 不可用。
  unproject 返回单位射线，点 = ray * depth。
深度真值（激光投影）与 LSS 的深度桶必须用同一个定义，见 `depth_type`。
"""
import torch

from ..registry import CAMERA_MODELS


class CameraModel:
    depth_type = "z"
    num_distortion = 0

    def unproject(self, uv, intrinsics, distortion=None):
        """uv (..., 2) -> rays (..., 3), valid (...)"""
        raise NotImplementedError

    def project(self, pts, intrinsics, distortion=None):
        """pts (..., 3) 相机系 -> uv (..., 2), depth (...), valid (...)"""
        raise NotImplementedError


@CAMERA_MODELS.register_module()
class Pinhole(CameraModel):
    """可选 OpenCV radial-tangential 畸变 (k1, k2, p1, p2, k3)。图像已去畸变时 distortion 传 None 或全 0。"""

    depth_type = "z"
    num_distortion = 5

    def __init__(self, undistort_iters: int = 8):
        self.undistort_iters = undistort_iters

    @staticmethod
    def _distort(x, y, d):
        k1, k2, p1, p2, k3 = d.unbind(-1)
        r2 = x * x + y * y
        radial = 1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
        xd = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
        yd = y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
        return xd, yd

    def unproject(self, uv, intrinsics, distortion=None):
        fx, fy, cx, cy = intrinsics.unbind(-1)
        xd = (uv[..., 0] - cx) / fx
        yd = (uv[..., 1] - cy) / fy
        x, y = xd, yd
        if distortion is not None and distortion.shape[-1] >= 5:
            d = distortion[..., :5]
            for _ in range(self.undistort_iters):
                xr, yr = self._distort(x, y, d)
                x = x - (xr - xd)
                y = y - (yr - yd)
        rays = torch.stack([x, y, torch.ones_like(x)], dim=-1)
        valid = torch.isfinite(rays).all(-1)
        return rays, valid

    def project(self, pts, intrinsics, distortion=None, eps: float = 1e-3):
        fx, fy, cx, cy = intrinsics.unbind(-1)
        z = pts[..., 2]
        valid = z > eps
        zc = z.clamp(min=eps)
        x = pts[..., 0] / zc
        y = pts[..., 1] / zc
        if distortion is not None and distortion.shape[-1] >= 5:
            x, y = self._distort(x, y, distortion[..., :5])
        uv = torch.stack([fx * x + cx, fy * y + cy], dim=-1)
        return uv, z, valid


@CAMERA_MODELS.register_module()
class KannalaBrandt(CameraModel):
    """theta_d = theta * (1 + k1 θ² + k2 θ⁴ + k3 θ⁶ + k4 θ⁸)，与 OpenCV fisheye 一致。"""

    depth_type = "range"
    num_distortion = 4

    def __init__(self, max_theta: float = 1.70, newton_iters: int = 10):
        self.max_theta = max_theta
        self.newton_iters = newton_iters

    @staticmethod
    def _poly(theta, d):
        k1, k2, k3, k4 = d.unbind(-1)
        t2 = theta * theta
        theta_d = theta * (1 + t2 * (k1 + t2 * (k2 + t2 * (k3 + t2 * k4))))
        dtheta_d = 1 + t2 * (3 * k1 + t2 * (5 * k2 + t2 * (7 * k3 + t2 * 9 * k4)))
        return theta_d, dtheta_d

    def project(self, pts, intrinsics, distortion=None, eps: float = 1e-6):
        fx, fy, cx, cy = intrinsics.unbind(-1)
        if distortion is None:
            distortion = pts.new_zeros(pts.shape[:-1] + (4,))
        x, y, z = pts.unbind(-1)
        r = torch.sqrt(x * x + y * y)
        theta = torch.atan2(r, z)
        theta_d, _ = self._poly(theta, distortion[..., :4])
        scale = torch.where(r > eps, theta_d / r.clamp(min=eps), 1.0 / z.clamp(min=eps))
        uv = torch.stack([fx * x * scale + cx, fy * y * scale + cy], dim=-1)
        depth = torch.sqrt(x * x + y * y + z * z)
        valid = (theta < self.max_theta) & (depth > 1e-3)
        return uv, depth, valid

    def unproject(self, uv, intrinsics, distortion=None, eps: float = 1e-8):
        fx, fy, cx, cy = intrinsics.unbind(-1)
        mx = (uv[..., 0] - cx) / fx
        my = (uv[..., 1] - cy) / fy
        if distortion is None:
            distortion = mx.new_zeros(mx.shape + (4,))
        d = distortion[..., :4]
        theta_d = torch.sqrt(mx * mx + my * my)
        theta = theta_d.clone()
        for _ in range(self.newton_iters):
            f, df = self._poly(theta, d)
            theta = (theta - (f - theta_d) / df.clamp(min=1e-6)).clamp(0.0, 3.1)
        _, df = self._poly(theta, d)
        s = torch.where(theta_d > eps, torch.sin(theta) / theta_d.clamp(min=eps), torch.ones_like(theta))
        rays = torch.stack([mx * s, my * s, torch.cos(theta)], dim=-1)
        valid = (theta < self.max_theta) & (df > 0) & torch.isfinite(rays).all(-1)
        return rays, valid


def transform_points(T: torch.Tensor, pts: torch.Tensor) -> torch.Tensor:
    """T (..., 4, 4) 作用于 pts (..., 3)，前缀需可广播。"""
    return (T[..., :3, :3] @ pts.unsqueeze(-1)).squeeze(-1) + T[..., :3, 3]
