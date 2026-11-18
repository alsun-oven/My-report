import torch

from ..registry import MATCHERS


def _linear_sum_assignment(cost: torch.Tensor):
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError as e:
        raise ImportError("Hungarian matching needs scipy: pip install scipy") from e
    r, c = linear_sum_assignment(cost.detach().float().cpu().numpy())
    return torch.as_tensor(r, dtype=torch.long), torch.as_tensor(c, dtype=torch.long)


def focal_cls_cost(logits, gt_labels, alpha=0.25, gamma=2.0, eps=1e-8):
    """logits (Q, K)，gt_labels (M,) -> (Q, M)，DETR 系列的 focal 分类代价。"""
    p = logits.sigmoid()
    neg = (1 - alpha) * p ** gamma * -(1 - p + eps).log()
    pos = alpha * (1 - p) ** gamma * -(p + eps).log()
    return pos[:, gt_labels] - neg[:, gt_labels]


@MATCHERS.register_module()
class HungarianMatcher3D:
    """3D 框：focal 分类代价 + 归一化中心 L1 代价。"""

    def __init__(self, cls_weight: float = 2.0, center_weight: float = 0.25, alpha=0.25, gamma=2.0):
        self.cls_weight = cls_weight
        self.center_weight = center_weight
        self.alpha, self.gamma = alpha, gamma

    @torch.no_grad()
    def __call__(self, cls_logits, pred_center_norm, gt_labels, gt_center_norm):
        """cls_logits (Q, K)，pred_center_norm (Q, 2) ∈[0,1]，gt_center_norm (M, 2)。"""
        if gt_labels.numel() == 0:
            empty = torch.empty(0, dtype=torch.long)
            return empty, empty
        c_cls = focal_cls_cost(cls_logits.float(), gt_labels, self.alpha, self.gamma)
        c_ctr = torch.cdist(pred_center_norm.float(), gt_center_norm.float(), p=1)
        return _linear_sum_assignment(self.cls_weight * c_cls + self.center_weight * c_ctr / 0.05)


@MATCHERS.register_module()
class SlotMatcher:
    """车位：存在性 focal 代价 + 四角米制 L1（角点顺序由标注约定：入口左、入口右、里侧右、里侧左）。"""

    def __init__(self, exist_weight: float = 2.0, corner_weight: float = 1.0):
        self.exist_weight = exist_weight
        self.corner_weight = corner_weight

    @torch.no_grad()
    def __call__(self, exist_logits, pred_corners, gt_corners):
        """exist_logits (Q,)，pred_corners (Q, 4, 2)，gt_corners (M, 4, 2) 米。"""
        if gt_corners.shape[0] == 0:
            empty = torch.empty(0, dtype=torch.long)
            return empty, empty
        zeros = torch.zeros(gt_corners.shape[0], dtype=torch.long, device=exist_logits.device)
        c_exist = focal_cls_cost(exist_logits.float()[:, None], zeros)
        c_corner = torch.cdist(pred_corners.flatten(1).float(), gt_corners.flatten(1).float(), p=1) / 8.0
        return _linear_sum_assignment(self.exist_weight * c_exist + self.corner_weight * c_corner)
