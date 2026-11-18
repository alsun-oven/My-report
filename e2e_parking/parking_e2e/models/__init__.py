# 逐个 import，触发各文件里的 @register_module()
from . import backbones, bev_encoders, fusion, heads, losses, matchers, sensor_encoders, view_transforms
from .detector import DEFAULT_LOSS_WEIGHTS, ParkingE2E

__all__ = [
    "backbones", "bev_encoders", "fusion", "heads", "losses", "matchers", "sensor_encoders",
    "view_transforms", "ParkingE2E", "DEFAULT_LOSS_WEIGHTS",
]
