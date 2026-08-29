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

    def forward(self, x, need_weights: bool = False):   # (B, N, C)
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.heads, c // self.heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)    # each (B, heads, N, head_dim)
        if need_weights:
            # Explicit softmax so the weights can be read. Materialises the
            # NxN matrix, so this is DIAGNOSTIC ONLY -- at 49k tokens it would
            # be 2.4e9 floats. Run it on a tile, never on the full grid.
            attn = (q @ k.transpose(-2, -1)) / ((c // self.heads) ** 0.5)
            attn = attn.softmax(dim=-1)
            out = (attn @ v).transpose(1, 2).reshape(b, n, c)
            return self.drop(self.proj(out)), attn.mean(dim=1)
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(b, n, c)
        return self.drop(self.proj(out)), None


class CrossAttention(nn.Module):
    """Queries from the satellite stream, keys/values from a context stream.

    Why this exists rather than concatenating channels
    --------------------------------------------------
    The streams have genuinely different native resolutions and cadences:
    INSAT TIR is 4 km/30 min, water vapour 8 km, ERA5 thermodynamics ~25 km
    hourly, terrain static at 90 m. Early fusion forces all of them onto the
    4 km analysis grid BEFORE the model sees anything, which upsamples ERA5
    by ~6x and spends 36 tokens representing what is physically one value --
    fabricating structure that is not in the data.

    Cross-attention lets the fine, fast stream QUERY the coarse, slow one at
    its own resolution. Cheap, because the key/value set is tiny: ~1,700 ERA5
    tokens against ~49,000 satellite tokens is 84M pairs per layer, about 3%
    of the 2.4B that spatial self-attention already costs.

    The weights are also the XAI product. "Which thermodynamic context
    mattered at this location" is exactly what the explainability panel is
    specified to show, and here it is a direct read-out rather than a
    post-hoc attribution method.
    """

    def __init__(self, dim: int, heads: int = 4, dropout: float = 0.0):
        super().__init__()
        if dim % heads:
            raise ValueError(f"dim {dim} not divisible by heads {heads}")
        self.heads = heads
        self.q = nn.Linear(dim, dim, bias=True)
        self.kv = nn.Linear(dim, dim * 2, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, ctx, need_weights: bool = False):
        """x (B, N, C) queries; ctx (B, M, C) context. M << N."""
        b, n, c = x.shape
        m = ctx.shape[1]
        h, hd = self.heads, c // self.heads

        q = self.q(x).reshape(b, n, h, hd).transpose(1, 2)          # (B,h,N,hd)
        kv = self.kv(ctx).reshape(b, m, 2, h, hd).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]                                          # (B,h,M,hd)

        if need_weights:
            # Explicit path: SDPA does not return weights, and the weights are
            # the XAI output. Only used for diagnostics, never in training.
            attn = (q @ k.transpose(-2, -1)) / (hd ** 0.5)
            attn = attn.softmax(dim=-1)
            out = attn @ v
            w = attn.mean(dim=1)                                     # (B,N,M)
        else:
            out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
            w = None
        out = out.transpose(1, 2).reshape(b, n, c)
        return self.drop(self.proj(out)), w


class DividedSpaceTimeBlock(nn.Module):
    """Attend over time, then over space, then mix channels.

    Pre-norm residuals throughout: each sub-layer adds a correction to the
    running representation rather than replacing it, which is what keeps
    deep stacks trainable.
    """

    def __init__(self, dim: int, heads: int = 4, mlp_ratio: float = 4.0,
                 dropout: float = 0.0, cross: bool = False):
        super().__init__()
        self.n1, self.time_attn = nn.LayerNorm(dim), Attention(dim, heads, dropout)
        self.n2, self.space_attn = nn.LayerNorm(dim), Attention(dim, heads, dropout)
        self.cross_attn = CrossAttention(dim, heads, dropout) if cross else None
        self.nc = nn.LayerNorm(dim) if cross else None
        self.n3, self.mlp = nn.LayerNorm(dim), Mlp(dim, mlp_ratio, dropout)

    def forward(self, x, t: int, hw: int, ctx=None, need_weights: bool = False):
        # (B, T*HW, C)
        b, n, c = x.shape

        # reshape (not view) throughout: the permutes below leave the tensor
        # non-contiguous, and view() would raise on the stride rather than
        # copying.
        # --- temporal: each spatial position attends along its own history
        xt = x.reshape(b, t, hw, c).permute(0, 2, 1, 3).reshape(b * hw, t, c)
        xt, _ = self.time_attn(self.n1(xt))
        x = x + xt.reshape(b, hw, t, c).permute(0, 2, 1, 3).reshape(b, n, c)

        # --- spatial: each frame attends within itself
        xs = x.reshape(b * t, hw, c)
        xs, sw = self.space_attn(self.n2(xs), need_weights=need_weights)
        self.last_space_weights = sw
        x = x + xs.reshape(b, n, c)

        # --- cross: the satellite stream queries the thermodynamic context
        w = None
        if self.cross_attn is not None and ctx is not None:
            xc, w = self.cross_attn(self.nc(x), ctx, need_weights=need_weights)
            x = x + xc

        return x + self.mlp(self.n3(x)), w


class SpatioTemporalBackbone(nn.Module):
    """(B, C_in, T, H, W) -> (B, dim, H/patch, W/patch), a single fused map.

    The temporal axis is pooled away at the end: the heads predict a
    sequence of future frames from one summary of the past, rather than
    decoding autoregressively. Simpler to train and it cannot drift, at the
    cost of not modelling dependence *between* its own output frames.
    """

    def __init__(self, in_channels: int, dim: int = 128, depth: int = 4,
                 heads: int = 4, patch: int = 4, max_frames: int = 8,
                 max_tokens: int = 4096, dropout: float = 0.0,
                 context_channels: int = 0, context_patch: int = 2,
                 max_context_tokens: int = 4096):
        super().__init__()
        self.dim, self.patch = dim, patch
        self.stem = nn.Conv3d(in_channels, dim, kernel_size=(1, patch, patch),
                              stride=(1, patch, patch))

        # Context stream: the coarse, slow modality (ERA5 thermodynamics) kept
        # at ITS OWN resolution rather than upsampled onto the 4 km grid.
        # A separate 2-D stem, because it has no temporal axis of its own --
        # one reanalysis snapshot per sample, broadcast across the satellite
        # frames it spans.
        self.context_channels = context_channels
        self.context_patch = context_patch
        if context_channels:
            self.ctx_stem = nn.Conv2d(context_channels, dim,
                                      kernel_size=context_patch, stride=context_patch)
            self.ctx_side = max(1, round(max_context_tokens ** 0.5))
            self.ctx_row = nn.Parameter(torch.zeros(1, self.ctx_side, dim))
            self.ctx_col = nn.Parameter(torch.zeros(1, self.ctx_side, dim))
            nn.init.trunc_normal_(self.ctx_row, std=0.02)
            nn.init.trunc_normal_(self.ctx_col, std=0.02)
            self.ctx_norm = nn.LayerNorm(dim)
        # 2-D SEPARABLE position embeddings, not a flat 1-D table.
        #
        # A flat table sliced [:hw] encodes token i as (i // wp, i % wp), and
        # wp changes with grid WIDTH. Train on 96x96 tiles (wp=24) and infer on
        # the 912x864 grid (wp=216) and index 24 means (row 1, col 0) in
        # training but (row 0, col 24) at inference. It does not raise -- it is
        # silently, spatially wrong, which is exactly the failure that would
        # have made "train on tiles, infer on the full grid" produce garbage.
        #
        # Separable row/column embeddings keep the meaning of a row index
        # independent of the width, and each axis interpolates cleanly on its
        # own when the inference grid is larger than the trained one.
        # exactly the trained side, so the native resolution needs NO
        # interpolation (int(sqrt)+1 gave 25 for a 24-token grid, resampling
        # 25->24 on every forward pass at the trained size)
        self.max_side = max(1, round(max_tokens ** 0.5))
        self.row_pos = nn.Parameter(torch.zeros(1, self.max_side, dim))
        self.col_pos = nn.Parameter(torch.zeros(1, self.max_side, dim))
        self.time_pos = nn.Parameter(torch.zeros(1, max_frames, 1, dim))
        nn.init.trunc_normal_(self.row_pos, std=0.02)
        nn.init.trunc_normal_(self.col_pos, std=0.02)
        nn.init.trunc_normal_(self.time_pos, std=0.02)
        self.blocks = nn.ModuleList(
            [DividedSpaceTimeBlock(dim, heads, dropout=dropout,
                                   cross=bool(context_channels))
             for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)

    def spatial_pos(self, hp: int, wp: int):
        """Position embedding for an (hp, wp) token grid.

        Axes are interpolated INDEPENDENTLY when the inference grid is larger
        than the trained one -- standard practice for running a ViT at a new
        resolution, and well posed here because each axis is 1-D.
        """
        row, col = self.row_pos, self.col_pos            # (1, S, C)
        if hp != row.shape[1]:
            row = torch.nn.functional.interpolate(
                row.transpose(1, 2), size=hp, mode="linear",
                align_corners=False).transpose(1, 2)
        if wp != col.shape[1]:
            col = torch.nn.functional.interpolate(
                col.transpose(1, 2), size=wp, mode="linear",
                align_corners=False).transpose(1, 2)
        # broadcast to the 2-D grid, then flatten in the SAME row-major order
        # the tokens use
        return (row[:, :, None, :] + col[:, None, :, :]).reshape(1, hp * wp, -1)

    def encode_context(self, ctx):
        """(B, C_ctx, H_ctx, W_ctx) -> (B, M, dim) context tokens."""
        if ctx.ndim != 4:
            raise ValueError(f"context must be (B, C, H, W), got {tuple(ctx.shape)}")
        z = self.ctx_stem(ctx)                       # (B, dim, h, w)
        b, c, h, w = z.shape
        row, col = self.ctx_row, self.ctx_col
        if h != row.shape[1]:
            row = torch.nn.functional.interpolate(
                row.transpose(1, 2), size=h, mode="linear",
                align_corners=False).transpose(1, 2)
        if w != col.shape[1]:
            col = torch.nn.functional.interpolate(
                col.transpose(1, 2), size=w, mode="linear",
                align_corners=False).transpose(1, 2)
        pos = (row[:, :, None, :] + col[:, None, :, :]).reshape(1, h * w, c)
        z = z.permute(0, 2, 3, 1).reshape(b, h * w, c) + pos
        return self.ctx_norm(z)

    def forward(self, x, context=None, need_weights: bool = False):
        if x.ndim != 5:
            raise ValueError(f"expected (B, C, T, H, W), got {tuple(x.shape)}")
        b, _, t, h, w = x.shape
        if h % self.patch or w % self.patch:
            raise ValueError(f"{h}x{w} not divisible by patch {self.patch}")

        x = self.stem(x)                         # (B, dim, T, h', w')
        _, c, t, hp, wp = x.shape
        hw = hp * wp
        if t > self.time_pos.shape[1]:
            raise ValueError(f"{t} frames exceeds max_frames {self.time_pos.shape[1]}")

        pos = self.spatial_pos(hp, wp)            # (1, hp*wp, C)
        x = x.permute(0, 2, 3, 4, 1).reshape(b, t, hw, c)
        x = x + pos.unsqueeze(1) + self.time_pos[:, :t]
        x = x.reshape(b, t * hw, c)

        ctx_tok = None
        if context is not None:
            if not self.context_channels:
                raise ValueError(
                    "context supplied but the backbone was built with "
                    "context_channels=0 -- cross-attention is not wired in")
            ctx_tok = self.encode_context(context)

        self.last_cross_weights = None
        weights = []
        for blk in self.blocks:
            x, w = blk(x, t, hw, ctx=ctx_tok, need_weights=need_weights)
            if w is not None:
                weights.append(w)
        if weights:
            # (layers, B, N, M) -- kept for the XAI panel: "which thermodynamic
            # context mattered where", read directly off the model rather than
            # reconstructed by a post-hoc attribution method
            self.last_cross_weights = torch.stack(weights)
        x = self.norm(x)

        # collapse time into one summary map
        x = x.reshape(b, t, hw, c).mean(dim=1)
        return x.transpose(1, 2).reshape(b, c, hp, wp)
