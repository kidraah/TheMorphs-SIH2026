"""Cross-satellite consistency: are 3D / 3DR / 3DS interchangeable?

The risk
--------
The Indian archive spans three instruments. Their channels are nominally
identical, but calibration offset, spectral response and noise differ between
satellites. Train across all three without checking and the model can learn
SATELLITE IDENTITY as a feature -- a shortcut that is stable in training,
invisible in the loss, and worthless operationally, because at inference
there is only ever one satellite.

That is the same class as the cloud/rain geolocation offset: a real,
learnable, wrong relationship that every score reports as success.

The measurement
---------------
INSAT-3DS came online 2024-05-17 and 3DR is still active, so there is an
overlap of coincident scenes. Compare brightness-temperature DISTRIBUTIONS
on scenes from the same timestamp: if the two instruments see the same
atmosphere, their distributions should differ only by noise.

Percentiles, not means: a mean difference can hide a calibration slope, and
the tails are where the convective signal lives.

Deciding
--------
  small   (< ~0.5 K across percentiles)  -> train across satellites freely
  medium  (~0.5-2 K)                     -> per-satellite normalisation
  large   (> ~2 K, or slope not offset)  -> restrict training to one satellite

`compare_distributions` returns the numbers; the call is a judgement, and
the thresholds above are a starting point, not a law.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

PERCENTILES = (1.0, 5.0, 10.0, 25.0, 50.0, 75.0, 90.0, 95.0, 99.0, 99.9)

# Below this, the p99.9 comparison is sampling noise rather than calibration.
# Measured on N(275, 20): two independent samples differ at p99.9 by 1.31 K at
# n=100k, 0.12 K at n=1M and 0.10 K at n=7.9M -- so a small crop can
# manufacture a "shape-differs" verdict out of nothing. A full INSAT TIR scan
# is ~7.9M pixels, so real comparisons are comfortably above this; a cropped
# region may not be.
MIN_RELIABLE_N = 500_000


@dataclass
class ChannelComparison:
    channel: str
    percentiles: tuple
    a_values: np.ndarray
    b_values: np.ndarray
    label_a: str
    label_b: str
    n_a: int
    n_b: int

    @property
    def underpowered(self) -> bool:
        """Too few pixels for the tail percentiles to mean anything."""
        return min(self.n_a, self.n_b) < MIN_RELIABLE_N

    @property
    def differences(self) -> np.ndarray:
        return self.b_values - self.a_values

    @property
    def max_abs_diff(self) -> float:
        return float(np.nanmax(np.abs(self.differences)))

    @property
    def median_diff(self) -> float:
        return float(np.nanmedian(self.differences))

    @property
    def is_offset_like(self) -> bool:
        """A constant offset is correctable by normalisation; a varying one
        (a calibration SLOPE) is not, and argues for one satellite only."""
        d = self.differences[np.isfinite(self.differences)]
        return d.size > 2 and float(np.nanstd(d)) < 0.5 * max(abs(self.median_diff), 0.1)

    @property
    def verdict(self) -> str:
        m = self.max_abs_diff
        if not np.isfinite(m):
            return "undetermined"
        if self.underpowered and m < 2.0:
            # Cannot separate a real difference from tail sampling noise.
            return "underpowered"
        if m < 0.5:
            return "consistent"
        if m < 2.0:
            return "offset" if self.is_offset_like else "shape-differs"
        return "large"

    def report(self) -> str:
        lines = [f"{self.channel}: {self.label_a} (n={self.n_a:,}) vs "
                 f"{self.label_b} (n={self.n_b:,})"]
        lines.append("   " + "  ".join(f"p{q:g}" for q in self.percentiles))
        lines.append("   " + "  ".join(f"{d:+.2f}" for d in self.differences))
        lines.append(f"   median diff {self.median_diff:+.2f} K, "
                     f"max |diff| {self.max_abs_diff:.2f} K -> {self.verdict.upper()}")
        if self.verdict == "underpowered":
            lines.append(f"   -> only {min(self.n_a, self.n_b):,} pixels; the tail "
                         f"percentiles are sampling noise below ~{MIN_RELIABLE_N:,}. "
                         f"Compare full scans, not crops -- a small sample can "
                         f"manufacture a difference that is not there.")
        elif self.verdict == "consistent":
            lines.append("   -> safe to train across these satellites")
        elif self.verdict == "offset":
            lines.append("   -> constant offset: per-satellite normalisation should fix it")
        elif self.verdict == "shape-differs":
            lines.append("   -> NOT a constant offset (a calibration slope or spectral "
                         "difference). Normalisation will not fix this; prefer a single "
                         "satellite for training.")
        else:
            lines.append("   -> large discrepancy. Do NOT mix these in training until "
                         "explained -- the model would learn satellite identity.")
        return "\n".join(lines)


@dataclass
class CrossSatelliteReport:
    comparisons: list = field(default_factory=list)

    @property
    def safe_to_mix(self) -> bool:
        return bool(self.comparisons) and all(
            c.verdict == "consistent" for c in self.comparisons)

    @property
    def needs_single_satellite(self) -> bool:
        return any(c.verdict in ("shape-differs", "large") for c in self.comparisons)

    @property
    def underpowered(self) -> bool:
        return any(c.verdict == "underpowered" for c in self.comparisons)

    def report(self) -> str:
        lines = ["cross-satellite consistency", ""]
        lines += [c.report() for c in self.comparisons]
        lines.append("")
        if self.underpowered:
            lines.append("VERDICT: INCONCLUSIVE -- too few pixels to separate "
                         "calibration from sampling noise. Re-run on full scans.")
        elif self.safe_to_mix:
            lines.append("VERDICT: consistent -- train across satellites.")
        elif self.needs_single_satellite:
            lines.append("VERDICT: restrict training to ONE satellite (3RIMG spans "
                         "2016-10-11 to present and covers every hindcast event "
                         "from 2017, so it is the natural choice).")
        else:
            lines.append("VERDICT: per-satellite normalisation, then re-check.")
        return "\n".join(lines)


def compare_distributions(a: dict, b: dict, label_a: str = "3DR",
                          label_b: str = "3DS",
                          percentiles=PERCENTILES) -> CrossSatelliteReport:
    """Compare per-channel BT distributions between two coincident scenes.

    `a` and `b` map channel name -> array. Grids need not match in shape:
    distributions are compared, not pixels, precisely because the two
    satellites view from different sub-satellite longitudes and a pixelwise
    difference would confound geometry with calibration.
    """
    rep = CrossSatelliteReport()
    for ch in sorted(set(a) & set(b)):
        va = np.asarray(a[ch], dtype=np.float64).ravel()
        vb = np.asarray(b[ch], dtype=np.float64).ravel()
        va, vb = va[np.isfinite(va)], vb[np.isfinite(vb)]
        if va.size < 100 or vb.size < 100:
            continue
        rep.comparisons.append(ChannelComparison(
            channel=ch, percentiles=tuple(percentiles),
            a_values=np.percentile(va, percentiles),
            b_values=np.percentile(vb, percentiles),
            label_a=label_a, label_b=label_b,
            n_a=int(va.size), n_b=int(vb.size)))
    return rep


def coincident_scenes(files_a, files_b, tolerance_min: float = 20.0) -> list:
    """Pair files from two satellites by acquisition time.

    Returns [(path_a, path_b, minutes_apart)]. Both run half-hourly but are
    not synchronised, so exact equality would find nothing.
    """
    from datetime import datetime
    from .insat import read_metadata

    def stamp(p):
        m = read_metadata(p)
        d, t = m.get("Acquisition_Date"), str(m.get("Acquisition_Time_in_GMT", "")).zfill(4)
        try:
            return datetime.strptime(f"{d} {t}", "%d%b%Y %H%M")
        except Exception:
            return None

    A = [(p, stamp(p)) for p in files_a]
    B = [(p, stamp(p)) for p in files_b]
    out = []
    for pa, ta in A:
        if ta is None:
            continue
        best, gap = None, None
        for pb, tb in B:
            if tb is None:
                continue
            g = abs((tb - ta).total_seconds()) / 60.0
            if gap is None or g < gap:
                best, gap = pb, g
        if best is not None and gap <= tolerance_min:
            out.append((pa, best, gap))
    return out
