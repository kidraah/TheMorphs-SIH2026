"""Generic value pile-up detector. Run on EVERY new data source before trusting it.

Three for three so far on this project, all the same shape:

  1. SEVIR VIL byte 255      -> decoded to 81.33 kg/m^2, the top of the range
  2. INSAT/SEVIR int16 min   -> decoded to -327.68 degC, the coldest cloud top
  3. INSAT count->K LUT clamp-> 180.09 K in TWO channels with different physics

None raised. All three decoded to plausible-looking EXTREMES, which is the
worst possible failure: an extreme is exactly what a severe-weather model is
built to notice, so a sentinel becomes the strongest signal in the dataset.
And each was found only by hand, after the data was already in use.

Assume IMERG and INSAT L1B each hide one more, and that any new source does
too. This module makes the check mechanical rather than a matter of someone
thinking to look.

What it flags
-------------
A value held by an anomalous fraction of pixels. Three signals, because mass
alone is not enough -- rain fields legitimately pile up at exactly 0.0:

  fraction   how much of the data sits on this one value
  isolation  how far it sits from the rest of the distribution, in IQRs.
             A sentinel is usually a lone spike separated by a wide gap;
             a legitimate mode is adjacent to its neighbours.
  extremity  whether it is the minimum or maximum. A pile-up at an extreme
             is far more suspicious than one in the middle, because that is
             where fill values and clamps land.

It REPORTS. It does not auto-drop anything: a legitimate zero-mode and a
sentinel look identical on mass alone, and deciding between them needs the
product documentation.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PileUp:
    value: float
    count: int
    fraction: float
    is_min: bool
    is_max: bool
    isolation_iqr: float        # gap to nearest other value, in IQRs
    suspicion: str              # "high" | "medium" | "low"
    note: str

    def __str__(self) -> str:
        where = "min" if self.is_min else ("max" if self.is_max else "interior")
        return (f"[{self.suspicion.upper():6s}] value={self.value:g} "
                f"{100 * self.fraction:.3f}% of pixels ({where}, "
                f"isolation {self.isolation_iqr:.1f} IQR) -- {self.note}")


@dataclass
class PileUpReport:
    pileups: list
    n_finite: int
    n_total: int

    @property
    def has_suspicious(self) -> bool:
        return any(p.suspicion in ("high", "medium") for p in self.pileups)

    def report(self, name: str = "") -> str:
        head = f"value pile-up scan{' -- ' + name if name else ''}"
        lines = [head, f"  {self.n_finite:,} finite of {self.n_total:,} values"]
        if not self.pileups:
            lines.append("  no anomalous pile-ups")
        else:
            lines += [f"  {p}" for p in self.pileups]
        if self.has_suspicious:
            lines.append("  -> CHECK THE PRODUCT DOCS before using this source. "
                         "A fill value decoded as a physical extreme is the "
                         "failure mode that does not raise.")
        return "\n".join(lines)


def detect_pileups(arr, min_fraction: float = 0.001, max_report: int = 6,
                   ignore_values: tuple = ()) -> PileUpReport:
    """Find single values holding an anomalous share of the data.

    `min_fraction` 0.001 catches the 0.33% LUT clamp found in real INSAT data;
    the SEVIR byte-255 case was 27%, and the int16 sentinel 7%.
    """
    a = np.asarray(arr)
    flat = a.ravel()
    finite = np.isfinite(flat)
    v = flat[finite].astype(np.float64)
    if v.size == 0:
        return PileUpReport([], 0, int(flat.size))

    uniq, counts = np.unique(v, return_counts=True)
    vmin, vmax = uniq[0], uniq[-1]

    out = []
    order = np.argsort(counts)[::-1]
    for i in order[: max_report * 4]:
        val, cnt = float(uniq[i]), int(counts[i])
        frac = cnt / v.size
        if frac < min_fraction or val in ignore_values:
            continue

        # Scale from the data EXCLUDING this value. Using the full IQR breaks
        # on fields that are mostly one value: a rain grid that is 85% zeros
        # has q1 == q3 == 0, so every gap divided by it looks astronomically
        # isolated and the legitimate dry mode is flagged as a sentinel.
        core = v[v != val]
        if core.size > 10:
            cq1, cq3 = np.percentile(core, [25, 75])
            scale = max(cq3 - cq1, float(np.std(core)) * 0.5, 1e-9)
        else:
            scale = max(float(np.std(v)), 1e-9)

        # gap to the nearest OTHER observed value
        j = int(i)
        left = uniq[j - 1] if j > 0 else None
        right = uniq[j + 1] if j + 1 < uniq.size else None
        gaps = [abs(val - x) for x in (left, right) if x is not None]
        isolation = (min(gaps) / scale) if gaps else float("inf")

        is_min, is_max = val == vmin, val == vmax
        extreme = is_min or is_max

        # Isolation is the real signal, not mass. A fill value sits in a gap;
        # a legitimate mode is continuous with its neighbours however large it
        # is. Mass only sharpens the judgement once isolation is established.
        if isolation > 3.0:
            s, note = "high", ("isolated spike separated from the distribution "
                               "-- the signature of a fill value or a clamp")
        elif extreme and isolation > 0.5:
            s, note = "high", "isolated spike at an extreme -- fill value or clamp"
        elif isolation > 1.0:
            s, note = "medium", "separated from the rest of the distribution"
        elif extreme and frac > 0.5:
            s, note = ("low", "dominant mode at the natural boundary -- legitimate "
                       "for a non-negative quantity (e.g. dry pixels at 0), but "
                       "confirm it is not ALSO the fill value")
        elif extreme:
            s, note = "low", "at an extreme but continuous with its neighbours"
        else:
            s, note = "low", "interior mode, likely legitimate"
        out.append(PileUp(val, cnt, frac, is_min, is_max, isolation, s, note))

    out.sort(key=lambda p: (-{"high": 2, "medium": 1, "low": 0}[p.suspicion],
                            -p.fraction))
    return PileUpReport(out[:max_report], int(v.size), int(flat.size))


def scan_source(arrays: dict, min_fraction: float = 0.001,
                ignore_values: tuple = ()) -> dict:
    """Scan every channel of a new source. Returns name -> PileUpReport."""
    return {k: detect_pileups(v, min_fraction, ignore_values=ignore_values)
            for k, v in arrays.items()}
