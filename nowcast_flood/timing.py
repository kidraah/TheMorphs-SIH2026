"""Kirpich time of concentration, and the peak-arrival time built from it.

    t_c [min] = 0.01947 * L^0.77 * S^-0.385     (L in metres)

which is Kirpich (1940) 0.0078 L_ft^0.77 S^-0.385 with the foot-to-metre
constant folded in (3.28084^0.77 = 2.4965; 0.0078 * 2.4965 = 0.01947).

WHERE THIS IS BEING USED OUTSIDE ITS DERIVATION
-----------------------------------------------
Kirpich fitted seven agricultural watersheds in Tennessee of 0.4 to 45
HECTARES -- 0.004 to 0.45 km^2. The sub-basins here start at the channel
threshold, typically 25 km^2, which is roughly **two orders of magnitude
larger** than anything in the original sample. That is an extrapolation, and
naming it is not optional: an unqualified "warning lead time is 47 minutes"
from a formula fitted to Tennessee farm plots is exactly the kind of
confident, out-of-domain number this project keeps catching in other
people's data.

`time_of_concentration` therefore returns the value AND a validity flag per
basin, and `TimingResult.report()` states how many basins are outside the
fitted range. Use it as a screening estimate and an ordering, not as a
defensible absolute number, until it is checked against observed IMD flood
arrival times.

Two more caveats kept in the open:
  * Kirpich assumes an unlined natural channel. Concrete or lined urban
    channels run roughly 0.4x, dense overland grass roughly 2x. Applying one
    number across a basin containing both is a real error, not a rounding
    one -- `channel_factor` exists to make the assumption explicit.
  * Zero slope gives an infinite t_c. Flat basins return nan, not a very
    large number that would sort to the end of a warning list and look like
    a calm basin.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Kirpich's fitted range, in km^2.
KIRPICH_MIN_KM2, KIRPICH_MAX_KM2 = 0.004, 0.45

# Derived, not transcribed. The literature quotes the metric constant as
# "0.0195", which is the exact value rounded to three figures and differs
# from it by 0.15%. Deriving it keeps the imperial original as the single
# source of truth and lets the equivalence be asserted exactly in the tests
# rather than to a tolerance chosen to make a rounded constant pass.
KIRPICH_C_IMPERIAL = 0.0078            # L in feet, t_c in minutes
FEET_PER_METRE = 1.0 / 0.3048
KIRPICH_C_METRIC = KIRPICH_C_IMPERIAL * FEET_PER_METRE ** 0.77   # 0.0194727

CHANNEL_FACTORS = {
    "natural": 1.0,      # unlined natural channel -- the fitted case
    "lined": 0.4,        # concrete / asphalt channel
    "overland_grass": 2.0,
}


@dataclass
class TimingResult:
    t_c_min: np.ndarray          # (B,) time of concentration, minutes
    t_peak_min: np.ndarray       # (B,) time to peak at the outlet, minutes
    within_fitted_range: np.ndarray   # (B,) bool
    channel_factor: float
    rain_duration_min: float

    def report(self) -> str:
        n = self.t_c_min.size
        out = int(np.sum(~self.within_fitted_range))
        fin = np.isfinite(self.t_peak_min)
        lines = [f"time to peak: median "
                 f"{np.nanmedian(self.t_peak_min[fin]) if fin.any() else float('nan'):.0f} min "
                 f"over {int(fin.sum())}/{n} basins"]
        if out:
            lines.append(
                f"!!  {out}/{n} basins ({100*out/n:.0f}%) are outside Kirpich's "
                f"fitted range ({KIRPICH_MIN_KM2}-{KIRPICH_MAX_KM2} km^2). These "
                f"are screening estimates, not defensible lead times.")
        flat = int(np.sum(~np.isfinite(self.t_c_min)))
        if flat:
            lines.append(f"    {flat} basin(s) have undefined t_c (zero or "
                         f"unknown slope) and are nan, not large.")
        return "\n".join(lines)


def time_to_peak(tc_min, rain_duration_min: float) -> np.ndarray:
    """SCS lag: T_p = D/2 + 0.6 t_c, minutes.

    This -- not t_c -- is the number to quote as warning lead time: it is the
    delay between the rain falling and the peak reaching the basin outlet.
    """
    tc = np.asarray(tc_min, dtype=np.float64)
    return 0.5 * float(rain_duration_min) + 0.6 * tc


def time_of_concentration(length_m, slope, area_km2=None,
                          channel: str = "natural",
                          rain_duration_min: float = 0.0) -> TimingResult:
    """Kirpich t_c in minutes, with an out-of-range flag per basin.

    `t_peak_min` is filled here rather than left to the caller: an earlier
    version stored t_c in that field, and `report()` then printed t_c under
    the label "time to peak" -- 136 min where the true figure was 111. A
    quantity in a field named after a different quantity is the plainest
    possible way to publish a wrong warning time.
    """
    if channel not in CHANNEL_FACTORS:
        raise ValueError(f"channel must be one of {sorted(CHANNEL_FACTORS)}")
    k = CHANNEL_FACTORS[channel]
    L = np.asarray(length_m, dtype=np.float64)
    S = np.asarray(slope, dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        tc = k * KIRPICH_C_METRIC * np.power(L, 0.77) * np.power(S, -0.385)
    tc = np.where((L > 0) & (S > 0) & np.isfinite(S), tc, np.nan)

    if area_km2 is None:
        ok = np.ones(tc.shape, dtype=bool)
    else:
        a = np.asarray(area_km2, dtype=np.float64)
        ok = (a >= KIRPICH_MIN_KM2) & (a <= KIRPICH_MAX_KM2)
    return TimingResult(tc, time_to_peak(tc, rain_duration_min), ok, k,
                        float(rain_duration_min))


def peak_discharge(runoff_mm, area_km2, t_peak_min) -> np.ndarray:
    """SCS triangular unit hydrograph peak, m^3/s.

        q_p = 0.208 * A * Q / T_p        A km^2, Q mm, T_p hours

    Returns nan where T_p is undefined rather than a large discharge, for
    the same reason as above.
    """
    A = np.asarray(area_km2, dtype=np.float64)
    Q = np.asarray(runoff_mm, dtype=np.float64)
    Tp_h = np.asarray(t_peak_min, dtype=np.float64) / 60.0
    with np.errstate(invalid="ignore", divide="ignore"):
        q = 0.208 * A * Q / Tp_h
    return np.where(np.isfinite(Tp_h) & (Tp_h > 0), q, np.nan)
