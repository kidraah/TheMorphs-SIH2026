"""Task heads. Two geometries, because the labels have two geometries.

This is where docs/LABELS.md shows up in the architecture. The rain-rate and
extreme-rain heads are supervised by gridded IMERG, so they decode to a
grid. The cloudburst head is supervised by IMD AWS/ARG station reports at
irregular points, so it must produce a value AT each station -- not a grid
that gets sampled afterwards as an afterthought.

Sampling the shared feature map at station coordinates keeps the gradient
path honest: the loss at a station updates exactly the features under that
station, with bilinear weights, and nothing else.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GridHead(nn.Module):
    """(B, dim, h, w) -> (B, lead, H, W) logits on the analysis grid."""

    def __init__(self, dim: int, lead_steps: int, patch: int,
                 hidden: int | None = None):
        super().__init__()
        hidden = hidden or dim
        self.lead_steps, self.patch = lead_steps, patch
        self.net = nn.Sequential(
            nn.Conv2d(dim, hidden, 3, padding=1), nn.GELU(),
            nn.Conv2d(hidden, lead_steps * patch * patch, 1),
        )

    def forward(self, feat):
        b, _, h, w = feat.shape
        x = self.net(feat)                                    # (B, L*p*p, h, w)
        # pixel shuffle back to full resolution: learned upsampling, no
        # interpolation artefacts baked in
        x = x.view(b, self.lead_steps, self.patch, self.patch, h, w)
        x = x.permute(0, 1, 4, 2, 5, 3).reshape(
            b, self.lead_steps, h * self.patch, w * self.patch)
        return x


class PointHead(nn.Module):
    """(B, dim, h, w) + station coords -> (B, lead, S) logits.

    coords are normalised to [-1, 1] as (x, y), matching grid_sample's
    convention: x indexes width, y indexes height. `normalise_coords` below
    converts pixel positions for you -- getting this order wrong silently
    transposes every station, which is the kind of bug that produces a model
    that trains fine and is worthless.
    """

    def __init__(self, dim: int, lead_steps: int, hidden: int = 128):
        super().__init__()
        self.lead_steps = lead_steps
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(),
            nn.Linear(hidden, lead_steps),
        )

    def forward(self, feat, coords):
        if coords.ndim != 3 or coords.shape[-1] != 2:
            raise ValueError(f"coords must be (B, S, 2) in [-1, 1], "
                             f"got {tuple(coords.shape)}")
        b, s, _ = coords.shape
        grid = coords.view(b, 1, s, 2)                       # (B, 1, S, 2)
        sampled = F.grid_sample(feat, grid, mode="bilinear",
                                align_corners=False)         # (B, dim, 1, S)
        sampled = sampled.squeeze(2).transpose(1, 2)         # (B, S, dim)
        return self.net(sampled).transpose(1, 2)             # (B, lead, S)


def normalise_coords(pixel_xy: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Pixel (x, y) -> grid_sample's [-1, 1], align_corners=False.

    Under align_corners=False a pixel CENTRE at index i maps to
    (2i + 1) / size - 1, so pixel 0 sits at -1 + 1/size rather than exactly
    -1. Using the align_corners=True convention here instead would offset
    every station by half a pixel -- 2 km on a 4 km grid, silently.
    """
    if pixel_xy.shape[-1] != 2:
        raise ValueError(f"expected (..., 2) as (x, y), got {tuple(pixel_xy.shape)}")
    x = (2.0 * pixel_xy[..., 0] + 1.0) / width - 1.0
    y = (2.0 * pixel_xy[..., 1] + 1.0) / height - 1.0
    return torch.stack([x, y], dim=-1)
