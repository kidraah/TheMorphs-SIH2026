"""Turning IMERG rain rates into the three Indian label heads.

Recap of the label decision (docs/LABELS.md), because the geometry differs
per head and that drives everything here:

    rain_rate      IMERG, gridded, a fixed rate threshold
    extreme_rain   IMERG, gridded, >99.9th percentile PER CELL PER MONTH
    cloudburst     IMD AWS/ARG station points, >100 mm/hr, POINT geometry

Why the extreme head is per-cell-per-month
------------------------------------------
A single all-India rain threshold would put nearly every positive in the
Western Ghats and the northeast in July and August, and almost none in the
Deccan interior or in winter. The model would then learn geography and
season -- which it can read off the DEM and the calendar -- instead of the
atmospheric state that distinguishes a storm day from a quiet one in the
same place at the same time of year.

A per-cell-per-month percentile asks the question that actually matters:
*is this extreme FOR HERE, NOW?* It also makes the base rate roughly uniform
in space, which is what stops the loss being dominated by a few wet cells.

Cost, stated plainly: the threshold is then not a fixed physical quantity, so
"extreme" means something different in Cherrapunji and in Jaisalmer. That is
the correct trade for a warning system -- a warning is relative to local
normality -- but it must be said out loud, and it is why the head is named
"extreme rain" rather than "cloudburst".

Climatology has to come from TRAIN YEARS ONLY
---------------------------------------------
The percentile is fitted data. Computing it over the whole record leaks
information about the validation and test periods into the labels the model
trains on, which inflates every score that follows.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# mm/hr. "Raining meaningfully" rather than drizzle; IMERG has skill at this
# rate where it does not at 0.1 mm/hr.
RAIN_RATE_MM_HR = 1.0

# The extreme head's percentile. 99.9 of half-hourly samples in a month is
# roughly one sample per two months per cell -- rare enough to be an event,
# frequent enough to be learnable.
EXTREME_PERCENTILE = 99.9

# IMD's operational cloudburst definition, for the POINT head only. It is not
# applied to IMERG: a 0.1deg cell is ~120 km^2 and averages a 20-30 km^2
# cloudburst below this threshold, which is the whole reason that head uses
# station data instead. See docs/LIMITATIONS.md #1.
CLOUDBURST_MM_HR = 100.0


@dataclass
class RainClimatology:
    """Per-cell, per-month percentile thresholds.

    thresholds[m] is the (H, W) threshold field for calendar month m (1-12).
    """
    thresholds: dict = field(default_factory=dict)
    percentile: float = EXTREME_PERCENTILE
    n_samples: dict = field(default_factory=dict)
    train_years: tuple = ()

    def threshold_for(self, month: int) -> np.ndarray:
        if month not in self.thresholds:
            raise KeyError(
                f"no climatology for month {month}; have "
                f"{sorted(self.thresholds)}. A month with no training data "
                f"cannot be labelled -- do not fall back to another month.")
        return self.thresholds[month]

    def save(self, path):
        np.savez_compressed(
            path, percentile=self.percentile,
            train_years=np.asarray(self.train_years),
            **{f"m{m}": t for m, t in self.thresholds.items()},
            **{f"n{m}": np.asarray(n) for m, n in self.n_samples.items()})

    @classmethod
    def load(cls, path):
        with np.load(path) as d:
            months = {int(k[1:]): d[k] for k in d.files if k.startswith("m")}
            counts = {int(k[1:]): int(d[k]) for k in d.files if k.startswith("n")}
            return cls(thresholds=months, percentile=float(d["percentile"]),
                       n_samples=counts,
                       train_years=tuple(d["train_years"].tolist()))


class ClimatologyAccumulator:
    """Streams IMERG granules into per-cell-per-month percentile thresholds.

    Percentiles need the distribution, and holding every half-hourly grid for
    a month in memory is not possible (a month is ~1450 grids). So rain rates
    are accumulated into per-cell HISTOGRAMS with logarithmic bins, and the
    percentile is read off the cumulative counts.

    Log bins because rain is heavily skewed: linear bins would put almost
    every sample in the first bin and resolve the tail -- the part we need --
    not at all.
    """

    def __init__(self, shape, n_bins: int = 256, max_mm_hr: float = 400.0):
        self.shape = tuple(shape)
        self.n_bins = int(n_bins)
        # bin i covers [edges[i], edges[i+1]); edges[0] = 0 exactly so dry
        # samples land in their own bin rather than being logged.
        self.edges = np.concatenate(
            [[0.0], np.logspace(np.log10(0.01), np.log10(max_mm_hr), n_bins - 1)])
        self.hist = {}
        self.counts = {}

    def add(self, rate: np.ndarray, month: int):
        r = np.asarray(rate, dtype=np.float64)
        if r.shape != self.shape:
            raise ValueError(f"expected {self.shape}, got {r.shape}")
        if month not in self.hist:
            self.hist[month] = np.zeros((self.n_bins,) + self.shape, dtype=np.int32)
            self.counts[month] = 0
        valid = np.isfinite(r)
        idx = np.clip(np.digitize(np.where(valid, r, 0.0), self.edges) - 1,
                      0, self.n_bins - 1)
        # scatter-add one count per valid cell into its bin
        flat = self.hist[month].reshape(self.n_bins, -1)
        np.add.at(flat, (idx.ravel(), np.arange(flat.shape[1])),
                  valid.ravel().astype(np.int32))
        self.counts[month] += 1

    def finalise(self, percentile: float = EXTREME_PERCENTILE,
                 train_years=()) -> RainClimatology:
        out = {}
        for m, h in self.hist.items():
            total = h.sum(axis=0)
            target = total * (percentile / 100.0)
            cum = np.cumsum(h, axis=0)
            # first bin whose cumulative count reaches the target
            reached = cum >= target[None, ...]
            first = np.argmax(reached, axis=0)
            # upper edge of that bin: conservative, so the threshold is not
            # below the percentile it claims
            thr = self.edges[np.minimum(first + 1, len(self.edges) - 1)]
            out[m] = np.where(total > 0, thr, np.nan).astype(np.float32)
        return RainClimatology(thresholds=out, percentile=percentile,
                               n_samples=dict(self.counts),
                               train_years=tuple(train_years))


def rain_labels(rate: np.ndarray, threshold_mm_hr: float = RAIN_RATE_MM_HR):
    """(label, valid) for the gridded rain-rate head."""
    r = np.asarray(rate, dtype=np.float64)
    valid = np.isfinite(r)
    return (np.where(valid, r, 0.0) > threshold_mm_hr).astype(np.float32), valid


def extreme_labels(rate: np.ndarray, clim: RainClimatology, month: int):
    """(label, valid) for the gridded extreme-rain head.

    Cells where the climatology is undefined (never any data in that month)
    are marked INVALID rather than negative: "we cannot say" is not "no
    event", and scoring them as negatives would silently reward the model for
    predicting nothing there.
    """
    r = np.asarray(rate, dtype=np.float64)
    thr = clim.threshold_for(month)
    if r.shape != thr.shape:
        raise ValueError(f"rate {r.shape} vs climatology {thr.shape}")
    valid = np.isfinite(r) & np.isfinite(thr)
    return (np.where(valid, r, 0.0) > np.where(valid, thr, np.inf)).astype(np.float32), valid


def cloudburst_labels(station_mm_hr: np.ndarray,
                      threshold: float = CLOUDBURST_MM_HR):
    """(label, valid) for the POINT head, from IMD station observations.

    Deliberately not derived from IMERG: a 0.1deg cell averages a
    20-30 km^2 cloudburst below 100 mm/hr, so an IMERG-derived "cloudburst"
    label would be mostly false negatives by construction.
    """
    v = np.asarray(station_mm_hr, dtype=np.float64)
    valid = np.isfinite(v)
    return (np.where(valid, v, 0.0) >= threshold).astype(np.float32), valid
