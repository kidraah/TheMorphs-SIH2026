"""Multi-task spatiotemporal nowcasting model.

One shared backbone, three heads, two output geometries -- the geometry
split follows directly from docs/LABELS.md, not from architectural taste.
"""
from .backbone import SpatioTemporalBackbone
from .heads import GridHead, PointHead, normalise_coords
from .losses import MultiTaskLoss, focal_loss_with_logits
from .model import DEFAULT_HEADS, HeadSpec, ModelConfig, MultiTaskNowcaster

__all__ = [
    "SpatioTemporalBackbone", "GridHead", "PointHead", "normalise_coords",
    "MultiTaskLoss", "focal_loss_with_logits",
    "MultiTaskNowcaster", "ModelConfig", "HeadSpec", "DEFAULT_HEADS",
]
