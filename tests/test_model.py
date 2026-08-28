"""Tests for the multi-task model skeleton.

Emphasis is on the things that fail SILENTLY: a point head sampling the
wrong location, a head detached from the shared backbone, logits mistaken
for probabilities, NaN targets poisoning the loss. All of those train
without error and produce a worthless model.
"""
import numpy as np
import pytest
import torch

from nowcast_model import (DEFAULT_HEADS, ModelConfig, MultiTaskLoss,
                           MultiTaskNowcaster, focal_loss_with_logits,
                           normalise_coords)
from nowcast_model.backbone import SpatioTemporalBackbone
from nowcast_model.heads import PointHead


def _model(**kw):
    kw.setdefault("in_channels", 2)
    kw.setdefault("context_frames", 2)
    kw.setdefault("grid_size", 32)
    kw.setdefault("lead_steps", 4)
    kw.setdefault("dim", 32)
    kw.setdefault("depth", 2)
    return MultiTaskNowcaster(ModelConfig(**kw))


# --------------------------------------------------------------------------
# Shapes and the two geometries
# --------------------------------------------------------------------------

def test_heads_emit_their_own_geometry():
    m = _model()
    x = torch.randn(2, 2, 2, 32, 32)
    coords = normalise_coords(torch.rand(2, 17, 2) * 31, 32, 32)
    out = m(x, coords)
    assert out["rain_rate"].shape == (2, 4, 32, 32)
    assert out["extreme_rain"].shape == (2, 4, 32, 32)
    assert out["cloudburst"].shape == (2, 4, 17)      # stations, not a grid


def test_default_heads_match_the_label_document():
    """docs/LABELS.md: there is deliberately no gridded 'cloudburst' head --
    IMD's 100 mm/hr over 20-30 km^2 cannot be expressed by a ~120 km^2 IMERG
    cell, so the gridded extreme head is named for what it can measure."""
    names = {h.name: h.geometry for h in DEFAULT_HEADS}
    assert names == {"rain_rate": "grid", "extreme_rain": "grid",
                     "cloudburst": "point"}


def test_point_head_without_coords_fails_loudly():
    m = _model()
    with pytest.raises(ValueError, match="needs station_coords"):
        m(torch.randn(1, 2, 2, 32, 32))


def test_backbone_rejects_indivisible_grid():
    b = SpatioTemporalBackbone(in_channels=2, dim=16, depth=1, patch=4)
    with pytest.raises(ValueError, match="not divisible"):
        b(torch.randn(1, 2, 2, 30, 30))


def test_backbone_rejects_wrong_rank():
    b = SpatioTemporalBackbone(in_channels=2, dim=16, depth=1, patch=4)
    with pytest.raises(ValueError, match=r"\(B, C, T, H, W\)"):
        b(torch.randn(1, 2, 32, 32))


# --------------------------------------------------------------------------
# The point head must sample the RIGHT place
# --------------------------------------------------------------------------

def test_normalise_coords_uses_the_align_corners_false_convention():
    """Pixel centre i maps to (2i+1)/size - 1.

    Using the align_corners=True convention instead would shift every
    station by half a pixel -- 2 km on a 4 km grid -- silently.
    """
    out = normalise_coords(torch.tensor([[[0.0, 0.0], [31.0, 31.0]]]), 32, 32)
    assert out[0, 0, 0].item() == pytest.approx(1 / 32 - 1)
    assert out[0, 1, 0].item() == pytest.approx(2 * 31 / 32 + 1 / 32 - 1)


def test_point_head_samples_the_named_pixel():
    """A spike at one pixel must be picked up by a station at that pixel and
    not by a station elsewhere. This is the test that catches an (x, y) /
    (row, col) transposition, which otherwise trains happily."""
    dim, h, w = 8, 16, 16
    feat = torch.zeros(1, dim, h, w)
    feat[0, :, 3, 11] = 10.0                     # row 3, col 11

    head = PointHead(dim, lead_steps=2)
    on = normalise_coords(torch.tensor([[[11.0, 3.0]]]), h, w)   # (x=col, y=row)
    off = normalise_coords(torch.tensor([[[3.0, 11.0]]]), h, w)  # transposed

    with torch.no_grad():
        a = head.net[0](torch.nn.functional.grid_sample(
            feat, on.view(1, 1, 1, 2), mode="bilinear",
            align_corners=False).squeeze(2).transpose(1, 2))
        b = head.net[0](torch.nn.functional.grid_sample(
            feat, off.view(1, 1, 1, 2), mode="bilinear",
            align_corners=False).squeeze(2).transpose(1, 2))
    assert a.abs().sum() > b.abs().sum() * 5, "coords must be (x, y), not (row, col)"


def test_point_head_rejects_malformed_coords():
    head = PointHead(8, lead_steps=2)
    with pytest.raises(ValueError, match=r"\(B, S, 2\)"):
        head(torch.zeros(1, 8, 4, 4), torch.zeros(1, 5, 3))


# --------------------------------------------------------------------------
# The backbone must actually be shared
# --------------------------------------------------------------------------

@pytest.mark.parametrize("head", ["rain_rate", "extreme_rain", "cloudburst"])
def test_every_head_sends_gradient_into_the_shared_backbone(head):
    """The point of multi-task learning. If one head were accidentally
    detached, it would still train its own layers and its loss would fall --
    while contributing nothing to the shared representation."""
    m = _model()
    x = torch.randn(2, 2, 2, 32, 32)
    coords = normalise_coords(torch.rand(2, 9, 2) * 31, 32, 32)

    out = m(x, coords)
    out[head].sum().backward()

    stem_grad = m.backbone.stem.weight.grad
    assert stem_grad is not None, f"{head} does not reach the backbone stem"
    assert stem_grad.abs().sum() > 0


def test_heads_do_not_share_their_own_parameters():
    m = _model()
    a = set(id(p) for p in m.heads["rain_rate"].parameters())
    b = set(id(p) for p in m.heads["extreme_rain"].parameters())
    assert not (a & b), "grid heads must be independent above the backbone"


# --------------------------------------------------------------------------
# Logits vs probabilities
# --------------------------------------------------------------------------

def test_forward_returns_logits_not_probabilities():
    """Feeding probabilities to a with-logits loss is a classic silent bug:
    it trains, badly, and nothing raises."""
    torch.manual_seed(0)
    m = _model()
    out = m(torch.randn(4, 2, 2, 32, 32) * 5,
            normalise_coords(torch.rand(4, 9, 2) * 31, 32, 32))
    allv = torch.cat([v.flatten() for v in out.values()])
    assert allv.min() < 0.0, "logits must be able to go negative"


def test_predict_proba_is_in_range_and_numpy():
    m = _model()
    p = m.predict_proba(torch.randn(2, 2, 2, 32, 32),
                        normalise_coords(torch.rand(2, 9, 2) * 31, 32, 32))
    for k, v in p.items():
        assert isinstance(v, np.ndarray)
        assert v.min() >= 0.0 and v.max() <= 1.0, k


def test_predict_proba_output_feeds_the_harness():
    """End to end: model output -> scorecard, in the shapes evaluate() wants."""
    from nowcast_eval import EvalConfig, evaluate

    m = _model()
    p = m.predict_proba(torch.randn(3, 2, 2, 32, 32),
                        normalise_coords(torch.rand(3, 9, 2) * 31, 32, 32))
    obs_grid = (np.random.default_rng(0).random(p["rain_rate"].shape) < 0.05).astype(float)
    r = evaluate(p["rain_rate"], obs_grid, EvalConfig(geometry="grid"))
    assert np.isfinite(r.pooled["headline"]["csi"])

    obs_pt = (np.random.default_rng(1).random(p["cloudburst"].shape) < 0.02).astype(float)
    rp = evaluate(p["cloudburst"], obs_pt, EvalConfig(geometry="point"))
    assert rp.pooled["fss"] == []


# --------------------------------------------------------------------------
# Loss
# --------------------------------------------------------------------------

def test_focal_loss_downweights_easy_examples():
    """The whole reason for focal loss at these base rates."""
    easy = torch.tensor([[8.0]])      # confident and correct
    hard = torch.tensor([[0.1]])      # uncertain
    y = torch.tensor([[1.0]])
    assert (focal_loss_with_logits(easy, y).item()
            < focal_loss_with_logits(hard, y).item() / 50)


def test_focal_loss_mask_excludes_missing_cells():
    """Satellite gaps decode to NaN. Unmasked they produce NaN gradients and
    silently destroy the run."""
    logits = torch.zeros(1, 4)
    targets = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    mask = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
    masked = focal_loss_with_logits(logits, targets, valid_mask=mask)
    assert torch.isfinite(masked)
    assert masked.item() == pytest.approx(
        focal_loss_with_logits(logits[:, :2], targets[:, :2]).item())


def test_multitask_loss_weights_shift_the_total():
    logits = {"a": torch.zeros(1, 4), "b": torch.zeros(1, 4)}
    targets = {"a": torch.ones(1, 4), "b": torch.zeros(1, 4)}
    base, _ = MultiTaskLoss()(logits, targets)
    up, _ = MultiTaskLoss(weights={"a": 5.0})(logits, targets)
    assert up.item() > base.item()


def test_multitask_loss_rejects_head_mismatch():
    with pytest.raises(ValueError, match="head mismatch"):
        MultiTaskLoss()({"a": torch.zeros(1, 2)}, {"b": torch.zeros(1, 2)})


def test_loss_reaches_every_head(capsys):
    m = _model()
    x = torch.randn(2, 2, 2, 32, 32)
    coords = normalise_coords(torch.rand(2, 9, 2) * 31, 32, 32)
    out = m(x, coords)
    targets = {"rain_rate": torch.rand(2, 4, 32, 32).round(),
               "extreme_rain": torch.rand(2, 4, 32, 32).round(),
               "cloudburst": torch.rand(2, 4, 9).round()}
    total, per_head = MultiTaskLoss()(out, targets)
    total.backward()
    assert set(per_head) == set(out)
    assert all(torch.isfinite(v) for v in per_head.values())
    assert m.backbone.stem.weight.grad.abs().sum() > 0
