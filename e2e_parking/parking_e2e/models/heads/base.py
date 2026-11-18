from typing import Any, Dict

import torch.nn as nn


class BaseHead(nn.Module):
    """所有头的统一接口。

    feats 是检测器汇总的特征字典：
      pv            {group: (B, N, C, h8, w8)}   图像特征（PV 头）
      depth_logits  {group: (B, N, D, h16, w16)} LSS 深度分布（重建头）
      depth_meta    {group: dict(dmin, step, num, stride, depth_type)}
      bev           (B, C, Y, X)                 融合 + 时序后的 BEV 特征
      voxel         (B, C, Z, Y, X)              GOD 体素特征
      nav           NavigationEncoder 的输出
      preds         {head_name: 该头 forward 的输出}，按头的顺序逐步填入（轨迹头读车位头的 token）
      sdf_fn        pts (B, N, 3) -> sdf (B, N)，参数冻结，供规划碰撞项和在线拒绝器使用
    """

    def forward(self, feats: Dict[str, Any]) -> Dict[str, Any]:
        return {}

    def loss(self, preds: Dict[str, Any], feats: Dict[str, Any], batch: Dict[str, Any]) -> Dict[str, Any]:
        """返回字典；以 loss 开头的项参与求和，其余项只记日志。"""
        raise NotImplementedError

    def predict(self, preds: Dict[str, Any], feats: Dict[str, Any], batch: Dict[str, Any]) -> Dict[str, Any]:
        return {}
