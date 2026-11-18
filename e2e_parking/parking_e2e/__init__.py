from . import data, engine, geometry, models  # noqa: F401  触发注册
from .config import Config, ConfigDict
from .registry import (BACKBONES, BEV_ENCODERS, CAMERA_MODELS, DATASETS, FUSERS, HEADS, LOSSES, MATCHERS,
                       MODELS, NECKS, OPTIMIZERS, SCHEDULERS, SENSOR_ENCODERS, TRANSFORMS, VIEW_TRANSFORMS,
                       Registry, build_from_cfg)

__version__ = "0.1.0"
