import inspect
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_with_defaults(registry, cfg, **defaults):
    """只把构造函数签名里存在的默认参数传进去，配置里显式写了的优先。

    用来把 grid、embed_dims 这类全局量只在模型顶层写一次。
    """
    if cfg is None:
        return None
    cls = registry.get(cfg["type"]) if isinstance(cfg["type"], str) else cfg["type"]
    if cls is None:
        return registry.build(cfg)
    params = inspect.signature(cls.__init__).parameters
    accepts_kwargs = any(p.kind == p.VAR_KEYWORD for p in params.values())
    extra = {k: v for k, v in defaults.items() if (k in params or accepts_kwargs) and k not in cfg}
    return registry.build(cfg, **extra)


def conv_bn_relu(cin, cout, k=3, s=1, dim=2):
    Conv = nn.Conv2d if dim == 2 else nn.Conv3d
    BN = nn.BatchNorm2d if dim == 2 else nn.BatchNorm3d
    return nn.Sequential(Conv(cin, cout, k, s, k // 2, bias=False), BN(cout), nn.ReLU(inplace=True))


class ResBlock(nn.Module):
    def __init__(self, cin, cout, stride=1, dim=2):
        super().__init__()
        Conv = nn.Conv2d if dim == 2 else nn.Conv3d
        BN = nn.BatchNorm2d if dim == 2 else nn.BatchNorm3d
        self.conv1 = Conv(cin, cout, 3, stride, 1, bias=False)
        self.bn1 = BN(cout)
        self.conv2 = Conv(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = BN(cout)
        self.down = None
        if stride != 1 or cin != cout:
            self.down = nn.Sequential(Conv(cin, cout, 1, stride, bias=False), BN(cout))

    def forward(self, x):
        idt = x if self.down is None else self.down(x)
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.bn2(self.conv2(x))
        return F.relu(x + idt, inplace=True)


class MLP(nn.Module):
    def __init__(self, cin, hidden, cout, num_layers=2, act=nn.ReLU):
        super().__init__()
        layers = []
        dims = [cin] + [hidden] * (num_layers - 1) + [cout]
        for i in range(num_layers):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < num_layers - 1:
                layers.append(act())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def fourier_encode(x: torch.Tensor, num_freqs: int) -> torch.Tensor:
    """x (..., D) -> (..., D * 2 * num_freqs)，频率 2^k * π。"""
    freqs = (2.0 ** torch.arange(num_freqs, device=x.device, dtype=x.dtype)) * math.pi
    xf = x.unsqueeze(-1) * freqs
    return torch.cat([xf.sin(), xf.cos()], dim=-1).flatten(-2)


def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(eps, 1 - eps)
    return torch.log(x / (1 - x))


class BEVDeformableAttention(nn.Module):
    """单尺度可变形注意力，纯 PyTorch（grid_sample）实现。

    ref 为 [0, 1] 归一化的 (x, y)，x 对应 BEV 宽度方向（NX）。偏移以格子为单位。
    """

    def __init__(self, embed_dims=256, num_heads=8, num_points=4):
        super().__init__()
        assert embed_dims % num_heads == 0
        self.h, self.p, self.d = num_heads, num_points, embed_dims
        self.sampling_offsets = nn.Linear(embed_dims, num_heads * num_points * 2)
        self.attention_weights = nn.Linear(embed_dims, num_heads * num_points)
        self.value_proj = nn.Linear(embed_dims, embed_dims)
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self._reset()

    def _reset(self):
        nn.init.zeros_(self.sampling_offsets.weight)
        thetas = torch.arange(self.h, dtype=torch.float32) * (2 * math.pi / self.h)
        init = torch.stack([thetas.cos(), thetas.sin()], -1)
        init = init / init.abs().max(-1, keepdim=True)[0]
        init = init.view(self.h, 1, 2).repeat(1, self.p, 1)
        for i in range(self.p):
            init[:, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias.copy_(init.view(-1))
        nn.init.zeros_(self.attention_weights.weight)
        nn.init.zeros_(self.attention_weights.bias)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.zeros_(self.value_proj.bias)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, query, ref, bev):
        B, Q, _ = query.shape
        _, C, Y, X = bev.shape
        H, P = self.h, self.p
        v = self.value_proj(bev.flatten(2).transpose(1, 2))                 # (B, YX, C)
        v = v.transpose(1, 2).reshape(B * H, C // H, Y, X)
        off = self.sampling_offsets(query).view(B, Q, H, P, 2)
        scale = query.new_tensor([X, Y])
        loc = ref[:, :, None, None, :] + off / scale                         # (B, Q, H, P, 2)
        w = self.attention_weights(query).view(B, Q, H, P).softmax(-1)
        grid = (loc * 2 - 1).permute(0, 2, 1, 3, 4).reshape(B * H, Q, P, 2)
        s = F.grid_sample(v, grid.to(v.dtype), mode="bilinear", padding_mode="zeros", align_corners=False)
        w = w.permute(0, 2, 1, 3).reshape(B * H, 1, Q, P)
        out = (s * w).sum(-1).view(B, H, C // H, Q).permute(0, 3, 1, 2).reshape(B, Q, C)
        return self.output_proj(out)


class DeformableDecoderLayer(nn.Module):
    def __init__(self, embed_dims=256, num_heads=8, num_points=4, ffn_dims=1024, dropout=0.0):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dims, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dims)
        self.cross_attn = BEVDeformableAttention(embed_dims, num_heads, num_points)
        self.norm2 = nn.LayerNorm(embed_dims)
        self.ffn = nn.Sequential(nn.Linear(embed_dims, ffn_dims), nn.ReLU(inplace=True),
                                 nn.Dropout(dropout), nn.Linear(ffn_dims, embed_dims))
        self.norm3 = nn.LayerNorm(embed_dims)

    def forward(self, q, qpos, ref, bev):
        h = q + qpos
        q = self.norm1(q + self.self_attn(h, h, q, need_weights=False)[0])
        q = self.norm2(q + self.cross_attn(q + qpos, ref, bev))
        return self.norm3(q + self.ffn(q))


class PositionEmbeddingLearned2D(nn.Module):
    def __init__(self, embed_dims, h, w):
        super().__init__()
        self.row = nn.Embedding(h, embed_dims // 2)
        self.col = nn.Embedding(w, embed_dims - embed_dims // 2)
        self.h, self.w = h, w

    def forward(self):
        y = self.row.weight[:, None, :].expand(-1, self.w, -1)
        x = self.col.weight[None, :, :].expand(self.h, -1, -1)
        return torch.cat([x, y], -1).view(self.h * self.w, -1)
