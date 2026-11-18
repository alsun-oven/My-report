from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..registry import LOSSES


def sigmoid_focal_loss(logits, targets, alpha=0.25, gamma=2.0, reduction="sum"):
    p = logits.sigmoid()
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce * (1 - p_t) ** gamma
    if alpha >= 0:
        loss = loss * (alpha * targets + (1 - alpha) * (1 - targets))
    return loss.sum() if reduction == "sum" else loss.mean() if reduction == "mean" else loss


def centernet_focal_loss(pred_logits, gt_heat, alpha=2.0, beta=4.0):
    """CornerNet / CenterNet 热图 focal loss，按正样本数归一化。"""
    pred = pred_logits.sigmoid().clamp(1e-4, 1 - 1e-4)
    pos = gt_heat.eq(1).float()
    neg = 1 - pos
    pos_loss = torch.log(pred) * (1 - pred) ** alpha * pos
    neg_loss = torch.log(1 - pred) * pred ** alpha * (1 - gt_heat) ** beta * neg
    num_pos = pos.sum().clamp(min=1)
    return -(pos_loss.sum() + neg_loss.sum()) / num_pos


def dice_loss(logits, target, num_classes, ignore_index=255, eps=1.0):
    """多类 soft dice。logits (B, K, ...)，target (B, ...)。"""
    valid = target != ignore_index
    tgt = torch.where(valid, target, torch.zeros_like(target))
    prob = logits.float().softmax(1)
    onehot = F.one_hot(tgt, num_classes).movedim(-1, 1).float()
    v = valid.unsqueeze(1).float()
    prob, onehot = prob * v, onehot * v
    dims = (0,) + tuple(range(2, prob.dim()))
    inter = (prob * onehot).sum(dims)
    union = prob.sum(dims) + onehot.sum(dims)
    present = onehot.sum(dims) > 0
    dice = 1 - (2 * inter + eps) / (union + eps)
    return dice[present].mean() if present.any() else logits.sum() * 0


@LOSSES.register_module()
class CrossEntropyDiceLoss(nn.Module):
    """分割用：按类加权的交叉熵 + Dice。车位线、轮挡像素少，class_weight 要给高。"""

    def __init__(self, num_classes: int, class_weight: Optional[Sequence[float]] = None,
                 ignore_index: int = 255, ce_weight: float = 1.0, dice_weight: float = 1.0):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        if class_weight is not None:
            self.register_buffer("class_weight", torch.tensor(class_weight, dtype=torch.float32), persistent=False)
        else:
            self.class_weight = None

    def forward(self, logits, target):
        ce = F.cross_entropy(logits.float(), target, weight=self.class_weight, ignore_index=self.ignore_index)
        if not torch.isfinite(ce):
            ce = logits.sum() * 0
        dl = dice_loss(logits, target, self.num_classes, self.ignore_index) if self.dice_weight > 0 else 0
        return self.ce_weight * ce + self.dice_weight * dl


@LOSSES.register_module()
class SigmoidFocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha, self.gamma = alpha, gamma

    def forward(self, logits, targets, avg_factor: float = 1.0):
        return sigmoid_focal_loss(logits.float(), targets.float(), self.alpha, self.gamma, "sum") / max(avg_factor, 1.0)
