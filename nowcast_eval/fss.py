"""Fractions Skill Score (Roberts & Lean, 2008, MWR 136:78-97).

The problem it solves: a pixel-wise CSI gives *zero credit* to a forecast
that puts a storm 20 km from where it landed -- it scores a miss and a
false alarm, worse than forecasting nothing at all. That is the wrong
signal for a convective nowcast, where an operator cares that a cell is
coming, not that it is coming to a particular 4 km pixel.

FSS compares the *fraction* of event pixels inside a neighbourhood of
each point, so a small displacement degrades the score smoothly instead
of destroying it. Sweeping the neighbourhood size tells you the spatial
scale at which the forecast becomes useful.

    FBS       = mean( (Pf - Po)^2 )
    FBS_worst = mean( Pf^2 ) + mean( Po^2 )       # zero overlap
    FSS       = 1 - FBS / FBS_worst

FSS = 1 is perfect, 0 is no skill. The conventional "useful" threshold is
FSS >= 0.5 + f0/2 where f0 is the domain event frequency.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import uniform_filter


def neighborhood_fractions(
    binary: np.ndarray,
    valid: np.ndarray,
    size: int,
) -> np.ndarray:
    """Fraction of VALID cells in each (size x size) window that are events.

    Masked and out-of-domain cells are excluded from both numerator and
    denominator rather than counted as no-event. Counting them as no-event
    is the usual bug here: it drags edge fractions toward zero and quietly
    inflates the score.
    """
    if size <= 1:
        out = np.where(valid, binary.astype(np.float64), np.nan)
        return out

    ndim = binary.ndim
    win = (1,) * (ndim - 2) + (size, size)  # filter the last two axes only

    v = valid.astype(np.float64)
    num = uniform_filter(np.where(valid, binary, 0).astype(np.float64),
                         size=win, mode="constant", cval=0.0)
    den = uniform_filter(v, size=win, mode="constant", cval=0.0)

    with np.errstate(invalid="ignore", divide="ignore"):
        frac = np.where(den > 0, num / den, np.nan)
    return frac


@dataclass(frozen=True)
class FSSResult:
    fss: float
    fbs: float
    fbs_worst: float
    neighborhood_px: int
    neighborhood_km: float
    threshold: float
    n_cells: int

    def to_dict(self) -> dict:
        return {
            "fss": self.fss,
            "fbs": self.fbs,
            "fbs_worst": self.fbs_worst,
            "neighborhood_px": self.neighborhood_px,
            "neighborhood_km": self.neighborhood_km,
            "threshold": self.threshold,
            "n_cells": self.n_cells,
        }


def fss(
    pred: np.ndarray,
    obs: np.ndarray,
    threshold: float,
    size: int,
    mask: np.ndarray | None = None,
    grid_km: float = float("nan"),
) -> FSSResult:
    """Pooled FSS over every sample in `pred` / `obs`.

    FBS and FBS_worst are accumulated across all samples before the ratio
    is taken, rather than averaging per-sample FSS values. Per-sample
    averaging lets a single quiet frame -- where both fields are empty and
    the score is undefined -- distort the aggregate.

    Arrays are (..., H, W); leading axes are samples / lead times.
    """
    pred = np.asarray(pred, dtype=np.float64)
    obs = np.asarray(obs, dtype=np.float64)
    if pred.shape != obs.shape:
        raise ValueError(f"shape mismatch: pred {pred.shape} vs obs {obs.shape}")
    if pred.ndim < 2:
        raise ValueError("need at least 2 spatial dims (..., H, W)")

    valid = np.isfinite(pred) & np.isfinite(obs)
    if mask is not None:
        valid = valid & np.asarray(mask, dtype=bool)

    pf = neighborhood_fractions(pred >= threshold, valid, size)
    po = neighborhood_fractions(obs > 0, valid, size)

    good = np.isfinite(pf) & np.isfinite(po) & valid
    n_cells = int(np.count_nonzero(good))
    if n_cells == 0:
        return FSSResult(float("nan"), float("nan"), float("nan"),
                         size, grid_km, float(threshold), 0)

    a = pf[good]
    b = po[good]
    fbs = float(np.mean((a - b) ** 2))
    fbs_worst = float(np.mean(a ** 2) + np.mean(b ** 2))

    # Both fields empty everywhere: perfectly agreed, but the ratio is 0/0.
    # nan is the honest answer -- a score of 1.0 here would let a model that
    # never fires look excellent on quiet days.
    score = float("nan") if fbs_worst == 0 else 1.0 - fbs / fbs_worst

    return FSSResult(score, fbs, fbs_worst, size, grid_km, float(threshold), n_cells)


def useful_scale_threshold(base_rate: float) -> float:
    """Roberts & Lean's 'useful forecast' line: FSS >= 0.5 + f0/2."""
    return 0.5 + base_rate / 2.0
