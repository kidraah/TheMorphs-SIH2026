"""2x2 contingency-table metrics.

    forecast yes/no  x  observed yes/no

        a = hits            b = false alarms
        c = misses          d = correct negatives

All of these are exact ratios of integer counts. There is nothing to get
subtly wrong in the arithmetic -- the risk is entirely in what goes into
the table (see config.py) and in reading the results without the base
rate beside them, which is why `base_rate` is a first-class field here.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _safe_div(num: float, den: float) -> float:
    """0/0 is undefined, not zero. Returning nan keeps that honest."""
    return float(num) / float(den) if den > 0 else float("nan")


@dataclass(frozen=True)
class ContingencyTable:
    hits: int
    false_alarms: int
    misses: int
    correct_negatives: int
    threshold: float

    # --- counts ------------------------------------------------------------
    @property
    def n(self) -> int:
        return self.hits + self.false_alarms + self.misses + self.correct_negatives

    @property
    def n_observed(self) -> int:
        return self.hits + self.misses

    @property
    def n_forecast(self) -> int:
        return self.hits + self.false_alarms

    @property
    def base_rate(self) -> float:
        """Fraction of samples where the event actually occurred.

        Report this next to CSI/FAR *always*. At a base rate of 1e-4 a
        forecast of 'no' everywhere scores 99.99% accuracy, and FAR
        becomes hypersensitive to a handful of false alarms.
        """
        return _safe_div(self.n_observed, self.n)

    # --- skill -------------------------------------------------------------
    @property
    def pod(self) -> float:
        """Probability of detection = hit rate = recall. a / (a + c)."""
        return _safe_div(self.hits, self.hits + self.misses)

    @property
    def far(self) -> float:
        """False alarm RATIO. b / (a + b).

        Not to be confused with the false alarm RATE, b / (b + d), which is
        the x-axis of a ROC curve. The literature uses both names for both
        quantities; this is the ratio, the one forecasters mean.
        """
        return _safe_div(self.false_alarms, self.hits + self.false_alarms)

    @property
    def false_alarm_rate(self) -> float:
        """b / (b + d) -- the ROC x-axis, NOT the false alarm ratio above.

        Distinct quantity, confusingly similar name. SEDI needs this one.
        """
        return _safe_div(self.false_alarms, self.false_alarms + self.correct_negatives)

    @property
    def success_ratio(self) -> float:
        """1 - FAR = precision."""
        far = self.far
        return float("nan") if np.isnan(far) else 1.0 - far

    @property
    def csi(self) -> float:
        """Critical success index (threat score). a / (a + b + c).

        Ignores correct negatives, which is what makes it the standard
        score for rare events -- accuracy is useless here.
        """
        return _safe_div(self.hits, self.hits + self.false_alarms + self.misses)

    @property
    def frequency_bias(self) -> float:
        """(a + b) / (a + c). >1 over-forecasting, <1 under-forecasting."""
        return _safe_div(self.n_forecast, self.n_observed)

    @property
    def ets(self) -> float:
        """Equitable threat score / Gilbert skill score.

        CSI corrected for the hits you would get by chance, which matters
        when comparing across regions or seasons with different base rates.
        """
        n = self.n
        if n == 0:
            return float("nan")
        hits_random = self.n_forecast * self.n_observed / n
        den = self.n_forecast + self.misses - hits_random
        return _safe_div(self.hits - hits_random, den)

    @property
    def sedi(self) -> float:
        """Symmetric Extremal Dependence Index (Ferro & Stephenson 2011).

        The rare-event metric. CSI degenerates toward zero as the base rate
        falls, whatever the forecast quality -- at 2e-4 it cannot separate
        "no skill" from "real skill on something genuinely rare", which
        makes it useless for the cloudburst head. SEDI is base-rate
        independent in the limit, so it stays interpretable there.

            SEDI = [ln F - ln H - ln(1-F) + ln(1-H)]
                 / [ln F + ln H + ln(1-F) + ln(1-H)]

        with H the hit rate (POD) and F the false alarm RATE (b/(b+d)).
        Range -1 to 1; 0 is no skill, 1 is perfect.

        Undefined (nan) when H or F hits 0 or 1 -- including for a perfect
        forecast, where F = 0. That is a real limitation, not a bug: with
        very few events a single threshold can push F to 0 and the score
        vanishes. Report it alongside the event count, and prefer a lower
        threshold over reading nan as failure.
        """
        h = self.pod
        f = self.false_alarm_rate
        if not (np.isfinite(h) and np.isfinite(f)):
            return float("nan")
        if h <= 0.0 or h >= 1.0 or f <= 0.0 or f >= 1.0:
            return float("nan")
        num = np.log(f) - np.log(h) - np.log(1 - f) + np.log(1 - h)
        den = np.log(f) + np.log(h) + np.log(1 - f) + np.log(1 - h)
        return float("nan") if den == 0 else float(num / den)

    @property
    def hss(self) -> float:
        """Heidke skill score: accuracy improvement over random chance."""
        a, b, c, d = self.hits, self.false_alarms, self.misses, self.correct_negatives
        den = (a + c) * (c + d) + (a + b) * (b + d)
        return _safe_div(2.0 * (a * d - b * c), den)

    def to_dict(self) -> dict:
        return {
            "threshold": self.threshold,
            "hits": self.hits,
            "false_alarms": self.false_alarms,
            "misses": self.misses,
            "correct_negatives": self.correct_negatives,
            "n": self.n,
            "base_rate": self.base_rate,
            "pod": self.pod,
            "far": self.far,
            "false_alarm_rate": self.false_alarm_rate,
            "sedi": self.sedi,
            "success_ratio": self.success_ratio,
            "csi": self.csi,
            "frequency_bias": self.frequency_bias,
            "ets": self.ets,
            "hss": self.hss,
        }


def contingency(
    pred: np.ndarray,
    obs: np.ndarray,
    threshold: float,
    mask: np.ndarray | None = None,
) -> ContingencyTable:
    """Build the table.

    pred : float probabilities in [0, 1]
    obs  : boolean / 0-1 truth, same shape
    mask : True where the cell is valid. Invalid cells are dropped, never
           silently counted as no-event.
    """
    pred = np.asarray(pred)
    obs = np.asarray(obs)
    if pred.shape != obs.shape:
        raise ValueError(f"shape mismatch: pred {pred.shape} vs obs {obs.shape}")

    valid = np.isfinite(pred) & np.isfinite(obs)
    if mask is not None:
        valid &= np.asarray(mask, dtype=bool)

    f = (pred >= threshold) & valid
    o = (obs > 0) & valid

    return ContingencyTable(
        hits=int(np.count_nonzero(f & o)),
        false_alarms=int(np.count_nonzero(f & ~o & valid)),
        misses=int(np.count_nonzero(~f & o & valid)),
        correct_negatives=int(np.count_nonzero(~f & ~o & valid)),
        threshold=float(threshold),
    )
