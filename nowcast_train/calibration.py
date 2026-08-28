"""Per-head operating point and probability calibration.

Two distinct problems, fixed in this order because only the first blocks
measurement:

1. **Operating point.** The harness's headline threshold of 0.5 is a
   deployment choice, not a property of the model. Measured on the SEVIR
   run: extreme_rain probabilities top out at 0.361 and cloudburst at 0.291,
   so at 0.5 both heads emit nothing, SEDI is undefined and checkpointing is
   blocked -- while their AUCs are 0.892 and 0.862, i.e. skill comparable to
   rain_rate's 0.887. The heads were never collapsed; the threshold was
   wrong. `select_threshold` picks it per head from validation data.

2. **Calibration.** Separately, the probabilities are badly scaled: mean
   predicted 0.0405 against a base rate of 6.25e-4, over-forecasting by
   ~65x, which is what BSS = -3.29 is reporting. Ranking is fine, magnitude
   is not. Isotonic regression fixes magnitude while preserving rank, so it
   cannot damage AUC or the threshold chosen above.

Isotonic is implemented here (pool-adjacent-violators) rather than pulled
from sklearn: it is ~40 lines, exact, and avoids adding a dependency for one
function.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from nowcast_eval.contingency import contingency


# --------------------------------------------------------------------------
# Operating point
# --------------------------------------------------------------------------

def select_threshold(probs: np.ndarray, targets: np.ndarray,
                     metric: str = "sedi",
                     candidates: np.ndarray | None = None,
                     mask: np.ndarray | None = None) -> tuple[float, float]:
    """Threshold maximising `metric` on VALIDATION data. Returns (thr, score).

    Candidates are drawn from the predicted distribution rather than a fixed
    grid, because a head whose probabilities live in [0.002, 0.36] shares no
    useful candidates with one spanning [0.04, 0.73].
    """
    p = np.asarray(probs, dtype=np.float64).ravel()
    t = np.asarray(targets, dtype=np.float64).ravel()
    if mask is not None:
        m = np.asarray(mask, dtype=bool).ravel()
        p, t = p[m], t[m]
    if candidates is None:
        qs = np.linspace(50.0, 99.99, 40)
        candidates = np.unique(np.percentile(p, qs))

    best, best_score = float("nan"), -np.inf
    for thr in candidates:
        c = contingency(p, t, float(thr))
        s = getattr(c, metric)
        if np.isfinite(s) and s > best_score:
            best, best_score = float(thr), float(s)
    return best, best_score


def select_thresholds(probs: dict, targets: dict, metric: str = "sedi") -> dict:
    """Per-head operating points. One threshold cannot serve all heads."""
    return {k: select_threshold(probs[k], targets[k], metric)[0] for k in probs}


# --------------------------------------------------------------------------
# Isotonic calibration
# --------------------------------------------------------------------------

def _pava(y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Pool adjacent violators: the monotone least-squares fit to y."""
    y = np.asarray(y, dtype=np.float64).copy()
    w = np.asarray(w, dtype=np.float64).copy()
    vals, wts, sizes = [], [], []
    for i in range(len(y)):
        v, ww, n = y[i], w[i], 1
        while vals and vals[-1] > v:                # violation -> pool
            pv, pw, pn = vals.pop(), wts.pop(), sizes.pop()
            v = (v * ww + pv * pw) / (ww + pw)
            ww, n = ww + pw, n + pn
        vals.append(v); wts.append(ww); sizes.append(n)
    out = np.empty_like(y)
    i = 0
    for v, n in zip(vals, sizes):
        out[i:i + n] = v
        i += n
    return out


@dataclass
class IsotonicCalibrator:
    """Monotone map from raw probability to observed frequency.

    Rank-preserving by construction, so AUC is unchanged and a threshold
    chosen before calibration maps through to the equivalent point after.
    """
    x: np.ndarray          # bin centres (raw probability)
    y: np.ndarray          # calibrated probability
    n_bins: int = 200

    @classmethod
    def fit(cls, probs, targets, n_bins: int = 200, mask=None) -> "IsotonicCalibrator":
        p = np.asarray(probs, dtype=np.float64).ravel()
        t = (np.asarray(targets).ravel() > 0).astype(np.float64)
        if mask is not None:
            m = np.asarray(mask, dtype=bool).ravel()
            p, t = p[m], t[m]

        # Quantile bins: equal-count, so rare high probabilities get their own
        # bins instead of being swamped by the mass near zero.
        edges = np.unique(np.percentile(p, np.linspace(0, 100, n_bins + 1)))
        if edges.size < 3:
            return cls(np.array([0.0, 1.0]), np.array([t.mean(), t.mean()]), n_bins)
        idx = np.clip(np.digitize(p, edges[1:-1]), 0, edges.size - 2)

        cx, cy, cw = [], [], []
        for b in range(edges.size - 1):
            sel = idx == b
            n = int(sel.sum())
            if n:
                cx.append(p[sel].mean()); cy.append(t[sel].mean()); cw.append(n)
        order = np.argsort(cx)
        cx = np.asarray(cx)[order]; cy = np.asarray(cy)[order]; cw = np.asarray(cw)[order]
        return cls(cx, _pava(cy, cw), n_bins)

    def transform(self, probs) -> np.ndarray:
        p = np.asarray(probs, dtype=np.float64)
        return np.interp(p, self.x, self.y, left=self.y[0], right=self.y[-1])

    def save(self, path):
        Path(path).write_text(json.dumps({"x": self.x.tolist(), "y": self.y.tolist(),
                                          "n_bins": self.n_bins}))

    @classmethod
    def load(cls, path):
        d = json.loads(Path(path).read_text())
        return cls(np.asarray(d["x"]), np.asarray(d["y"]), d["n_bins"])


def fit_calibrators(probs: dict, targets: dict, n_bins: int = 200) -> dict:
    return {k: IsotonicCalibrator.fit(probs[k], targets[k], n_bins) for k in probs}


def apply_calibrators(probs: dict, cals: dict) -> dict:
    return {k: (cals[k].transform(v) if k in cals else v) for k, v in probs.items()}
