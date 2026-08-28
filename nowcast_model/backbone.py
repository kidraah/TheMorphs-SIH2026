"""Shared spatiotemporal backbone.

What a transformer is doing here, in plain terms
------------------------------------------------
Attention lets every location in the input look at every other location and
decide how much each one matters. A convolution can only see a fixed small
window; attention can connect a moisture pool on one side of the domain to
a developing updraft on the other, and *learn* how strongly to connect them
rather than having that wired in.

The cost is that "everything looks at everything" grows quadratically. For
a 96x96 grid over 4 time steps that is 36,864 positions, and attending
across all of them at once is ~1.4 billion pairs per layer -- too slow and
too memory-hungry to train.

So attention is FACTORISED: each block attends over time first (each pixel
watches its own history), then over space (each frame looks at itself). Two
cheap operations instead of one impossibly expensive one, and stacking them
still lets information travel in both dimensions. This is the standard
"divided space-time attention" arrangement.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class Mlp(nn.Module):
    def __init__(self, dim: int, ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        hidden = int(dim * ratio)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, dim), nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    """Plain multi-head self-attention over whatever axis the caller folds in."""

    def __init__(self, dim: int, heads: int = 4, dropout: float = 0.0):
        super().__init__()
        if dim % heads:
            raise ValueError(f"dim {dim} not divisible by heads {heads}")
        self.heads = heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):                       # (B, N, C)
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.heads, c // self.heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)    # each (B, heads, N, head_dim)
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(b, n, c)
        return self.drop(self.proj(out))


class DividedSpaceTimeBlock(nn.Module):
    """Attend over time, then over space, then mix channels.

    Pre-norm residuals throughout: each sub-layer adds a correction to the
    running representation rather than replacing it, which is what keeps
    deep stacks trainable.
    """

    def __init__(self, dim: int, heads: int = 4, mlp_ratio: float = 4.0,
                 dropout: float = 0.0):
        super().__init__()
        self.n1, self.time_attn = nn.LayerNorm(dim), Attention(dim, heads, dropout)
        self.n2, self.space_attn = nn.LayerNorm(dim), Attention(dim, heads, dropout)
        self.n3, self.mlp = nn.LayerNorm(dim), Mlp(dim, mlp_ratio, dropout)

    def forward(self, x, t: int, hw: int):      # (B, T*HW, C)
        b, n, c = x.shape

        # reshape (not view) throughout: the permutes below leave the tensor
        # non-contiguous, and view() would raise on the stride rather than
        # copying.
        # --- temporal: each spatial position attends along its own history
        xt = x.reshape(b, t, hw, c).permute(0, 2, 1, 3).reshape(b * hw, t, c)
        xt = self.time_attn(self.n1(xt))
        x = x + xt.reshape(b, hw, t, c).permute(0, 2, 1, 3).reshape(b, n, c)

        # --- spatial: each frame attends within itself
        xs = x.reshape(b * t, hw, c)
        xs = self.space_attn(self.n2(xs))
        x = x + xs.reshape(b, n, c)

        return x + self.mlp(self.n3(x))


class SpatioTemporalBackbone(nn.Module):
    """(B, C_in, T, H, W) -> (B, dim, H/patch, W/patch), a single fused map.

    The temporal axis is pooled away at the end: the heads predict a
    sequence of future frames from one summary of the past, rather than
    decoding autoregressively. Simpler to train and it cannot drift, at the
    cost of not modelling dependence *between* its own output frames.
    """

    def __init__(self, in_channels: int, dim: int = 128, depth: int = 4,
                 heads: int = 4, patch: int = 4, max_frames: int = 8,
                 max_tokens: int = 4096, dropout: float = 0.0):
        super().__init__()
        self.dim, self.patch = dim, patch
        self.stem = nn.Conv3d(in_channels, dim, kernel_size=(1, patch, patch),
                              stride=(1, patch, patch))
        self.space_pos = nn.Parameter(torch.zeros(1, max_tokens, dim))
        self.time_pos = nn.Parameter(torch.zeros(1, max_frames, 1, dim))
        nn.init.trunc_normal_(self.space_pos, std=0.02)
        nn.init.trunc_normal_(self.time_pos, std=0.02)
        self.blocks = nn.ModuleList(
            [DividedSpaceTimeBlock(dim, heads, dropout=dropout) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):                        # (B, C, T, H, W)
        if x.ndim != 5:
            raise ValueError(f"expected (B, C, T, H, W), got {tuple(x.shape)}")
        b, _, t, h, w = x.shape
        if h % self.patch or w % self.patch:
            raise ValueError(f"{h}x{w} not divisible by patch {self.patch}")

        x = self.stem(x)                         # (B, dim, T, h', w')
        _, c, t, hp, wp = x.shape
        hw = hp * wp
        if hw > self.space_pos.shape[1]:
            raise ValueError(f"{hw} spatial tokens exceeds max_tokens "
                             f"{self.space_pos.shape[1]}")
        if t > self.time_pos.shape[1]:
            raise ValueError(f"{t} frames exceeds max_frames {self.time_pos.shape[1]}")

        x = x.permute(0, 2, 3, 4, 1).reshape(b, t, hw, c)
        x = x + self.space_pos[:, :hw].unsqueeze(1) + self.time_pos[:, :t]
        x = x.reshape(b, t * hw, c)

        for blk in self.blocks:
            x = blk(x, t, hw)
        x = self.norm(x)

        # collapse time into one summary map
        x = x.reshape(b, t, hw, c).mean(dim=1)
        return x.transpose(1, 2).reshape(b, c, hp, wp)
