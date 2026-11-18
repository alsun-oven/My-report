import math
from typing import Dict

import torch

from ..registry import OPTIMIZERS, SCHEDULERS

for _name in ("SGD", "Adam", "AdamW"):
    OPTIMIZERS.register_module(module=getattr(torch.optim, _name))


@SCHEDULERS.register_module()
class WarmupCosine(torch.optim.lr_scheduler.LambdaLR):
    """按迭代步：线性 warmup 后余弦衰减到 min_ratio。"""

    def __init__(self, optimizer, total_iters: int, warmup_iters: int = 500, min_ratio: float = 1e-2):
        def f(it):
            if it < warmup_iters:
                return (it + 1) / max(warmup_iters, 1)
            p = (it - warmup_iters) / max(total_iters - warmup_iters, 1)
            return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))

        super().__init__(optimizer, f)


def build_optimizer(model: torch.nn.Module, cfg: Dict):
    """cfg: dict(type="AdamW", lr=..., weight_decay=..., paramwise=dict(img_backbone=0.1))
    paramwise 里的键按参数名前缀匹配，值为学习率倍率。"""
    cfg = dict(cfg)
    paramwise = cfg.pop("paramwise", {}) or {}
    base_lr = cfg["lr"]
    groups: Dict[float, list] = {}
    no_decay: Dict[float, list] = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        mult = 1.0
        for prefix, m in paramwise.items():
            if name.startswith(prefix):
                mult = m
        target = no_decay if (p.ndim <= 1 or name.endswith(".bias")) else groups
        target.setdefault(mult, []).append(p)
    param_groups = [dict(params=v, lr=base_lr * m) for m, v in groups.items()]
    param_groups += [dict(params=v, lr=base_lr * m, weight_decay=0.0) for m, v in no_decay.items()]
    return OPTIMIZERS.build(dict(cfg, params=param_groups))
