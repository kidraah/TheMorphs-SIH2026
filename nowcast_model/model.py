"""The multi-task nowcaster: one backbone, three heads, two geometries."""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from .backbone import SpatioTemporalBackbone
from .heads import GridHead, PointHead


@dataclass
class HeadSpec:
    name: str
    geometry: str = "grid"          # "grid" | "point" -- must match EvalConfig
    lead_steps: int = 6

    def __post_init__(self):
        if self.geometry not in ("grid", "point"):
            raise ValueError(f"geometry must be 'grid' or 'point', got {self.geometry!r}")


# Mirrors docs/LABELS.md. Note there is no head called "cloudburst" on the
# grid: IMD's 100 mm/hr over 20-30 km^2 cannot be expressed by a ~120 km^2
# IMERG cell, so the gridded extreme head is named for what it can measure.
DEFAULT_HEADS = [
    HeadSpec("rain_rate", "grid"),
    HeadSpec("extreme_rain", "grid"),
    HeadSpec("cloudburst", "point"),
]


@dataclass
class ModelConfig:
    in_channels: int = 2            # ir107, ir069
    dim: int = 128
    depth: int = 4
    heads: int = 4
    patch: int = 4
    lead_steps: int = 6
    context_frames: int = 2
    grid_size: int = 96
    dropout: float = 0.0
    # Cross-attention fusion. 0 disables it and the model behaves exactly as
    # before (early fusion by channel concatenation).
    context_channels: int = 0       # e.g. 6 for cape/cin/tcwv/shear/conv/mconv
    context_patch: int = 2          # ERA5 ~25 km -> ~50 km tokens
    head_specs: list = field(default_factory=lambda: list(DEFAULT_HEADS))

    def __post_init__(self):
        for h in self.head_specs:
            h.lead_steps = self.lead_steps


class MultiTaskNowcaster(nn.Module):
    """(B, C, T, H, W) -> {head name: logits}.

    LOGITS, not probabilities. The loss functions expect logits for
    numerical stability; call `predict_proba` to get what the harness wants.
    Returning probabilities here and feeding them to a BCE-with-logits loss
    is a standard silent bug -- the model trains, badly, and nothing errors.
    """

    def __init__(self, config: ModelConfig | None = None):
        super().__init__()
        self.cfg = cfg = config or ModelConfig()
        max_tokens = (cfg.grid_size // cfg.patch) ** 2

        self.backbone = SpatioTemporalBackbone(
            in_channels=cfg.in_channels, dim=cfg.dim, depth=cfg.depth,
            heads=cfg.heads, patch=cfg.patch,
            max_frames=max(cfg.context_frames, 8),
            max_tokens=max(max_tokens, 4096), dropout=cfg.dropout,
            context_channels=cfg.context_channels,
            context_patch=cfg.context_patch)

        self.heads = nn.ModuleDict()
        self.geometries = {}
        for spec in cfg.head_specs:
            self.geometries[spec.name] = spec.geometry
            if spec.geometry == "grid":
                self.heads[spec.name] = GridHead(cfg.dim, spec.lead_steps, cfg.patch)
            else:
                self.heads[spec.name] = PointHead(cfg.dim, spec.lead_steps)

    @property
    def point_heads(self) -> list[str]:
        return [n for n, g in self.geometries.items() if g == "point"]

    def forward(self, x, station_coords=None, context=None,
                need_weights: bool = False):
        """context: (B, C_ctx, H_ctx, W_ctx) thermodynamic fields at their OWN
        resolution -- not upsampled to the satellite grid. None falls back to
        early fusion, i.e. whatever was concatenated into `x`."""
        feat = self.backbone(x, context=context, need_weights=need_weights)
        out = {}
        for name, head in self.heads.items():
            if self.geometries[name] == "point":
                if station_coords is None:
                    raise ValueError(
                        f"head {name!r} is point geometry and needs "
                        f"station_coords (B, S, 2) normalised to [-1, 1]; "
                        f"see heads.normalise_coords")
                out[name] = head(feat, station_coords)
            else:
                out[name] = head(feat)
        return out

    @property
    def cross_attention_weights(self):
        """(layers, B, N_satellite_tokens, M_context_tokens) from the last
        forward pass with need_weights=True. This IS the XAI product: which
        thermodynamic context the model consulted, per location."""
        return getattr(self.backbone, "last_cross_weights", None)

    @torch.no_grad()
    def predict_proba(self, x, station_coords=None) -> dict:
        """What the harness consumes: probabilities in [0, 1]."""
        self.eval()
        return {k: torch.sigmoid(v).cpu().numpy()
                for k, v in self.forward(x, station_coords).items()}

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
