"""Scores that use the raw probabilities, with no threshold applied.

These matter because thresholded scores throw away most of what the model
said. A model that outputs 0.51 everywhere and a model that outputs
confident, well-separated probabilities can have identical CSI at 0.5 and
be worth completely different amounts to an operator deciding whether to
evacuate a valley.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ReliabilityCurve:
    bin_edges: np.ndarray
    mean_forecast: np.ndarray   # mean predicted probability in each bin
    observed_freq: np.ndarray   # observed event frequency in each bin
    counts: np.ndarray

    def to_dict(self) -> dict:
        return {
            "bin_edges": self.bin_edges.tolist(),
            "mean_forecast": self.mean_forecast.tolist(),
            "observed_freq": self.observed_freq.tolist(),
            "counts": self.counts.tolist(),
        }


@dataclass(frozen=True)
class ProbabilisticScores:
    brier: float
    brier_climatology: float
    bss: float
    base_rate: float
    n: int
    reliability: ReliabilityCurve

    def to_dict(self) -> dict:
        return {
            "brier": self.brier,
            "brier_climatology": self.brier_climatology,
            "bss": self.bss,
            "base_rate": self.base_rate,
            "n": self.n,
            "reliability": self.reliability.to_dict(),
        }


def brier_score(pred: np.ndarray, obs: np.ndarray,
                mask: np.ndarray | None = None) -> float:
    """Mean squared error of the probability forecast. Lower is better."""
    pred = np.asarray(pred, dtype=np.float64)
    obs = np.asarray(obs, dtype=np.float64)
    valid = np.isfinite(pred) & np.isfinite(obs)
    if mask is not None:
        valid &= np.asarray(mask, dtype=bool)
    if not valid.any():
        return float("nan")
    return float(np.mean((pred[valid] - (obs[valid] > 0)) ** 2))


def reliability_curve(pred, obs, n_bins=10, mask=None) -> ReliabilityCurve:
    """Are the probabilities calibrated? Of everything called 30%, did ~30% happen?

    A model can rank events perfectly and still be badly calibrated. This is
    the plot that tells a disaster-management authority whether a stated 70%
    means anything.
    """
    pred = np.asarray(pred, dtype=np.float64)
    obs = np.asarray(obs, dtype=np.float64)
    valid = np.isfinite(pred) & np.isfinite(obs)
    if mask is not None:
        valid &= np.asarray(mask, dtype=bool)

    p = pred[valid]
    o = (obs[valid] > 0).astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)

    mean_f = np.full(n_bins, np.nan)
    obs_f = np.full(n_bins, np.nan)
    counts = np.zeros(n_bins, dtype=np.int64)

    if p.size:
        # right-closed on the final bin so p == 1.0 is not dropped
        idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)
        for b in range(n_bins):
            sel = idx == b
            counts[b] = int(np.count_nonzero(sel))
            if counts[b]:
                mean_f[b] = float(p[sel].mean())
                obs_f[b] = float(o[sel].mean())

    return ReliabilityCurve(edges, mean_f, obs_f, counts)


def probabilistic_scores(pred, obs, n_bins=10, mask=None) -> ProbabilisticScores:
    """Brier, Brier skill score against climatology, and the reliability curve.

    The reference forecast is the sample climatology -- predicting the base
    rate everywhere, every time. BSS = 0 means the model has learned nothing
    beyond how often the event happens. That is the bar to clear, and a
    surprising number of rare-event models sit below it.
    """
    pred = np.asarray(pred, dtype=np.float64)
    obs = np.asarray(obs, dtype=np.float64)
    valid = np.isfinite(pred) & np.isfinite(obs)
    if mask is not None:
        valid &= np.asarray(mask, dtype=bool)

    n = int(np.count_nonzero(valid))
    if n == 0:
        empty = ReliabilityCurve(np.linspace(0, 1, n_bins + 1),
                                 np.full(n_bins, np.nan),
                                 np.full(n_bins, np.nan),
                                 np.zeros(n_bins, dtype=np.int64))
        return ProbabilisticScores(float("nan"), float("nan"), float("nan"),
                                   float("nan"), 0, empty)

    o = (obs[valid] > 0).astype(np.float64)
    base = float(o.mean())
    bs = float(np.mean((pred[valid] - o) ** 2))
    bs_clim = float(np.mean((base - o) ** 2))  # == base * (1 - base)
    bss = float("nan") if bs_clim == 0 else 1.0 - bs / bs_clim

    return ProbabilisticScores(
        brier=bs,
        brier_climatology=bs_clim,
        bss=bss,
        base_rate=base,
        n=n,
        reliability=reliability_curve(pred, obs, n_bins=n_bins, mask=mask),
    )
