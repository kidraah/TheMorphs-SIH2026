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


# Fitted area ranges, km^2. These are the reason `method` exists.
FITTED_RANGE_KM2 = {
    "kirpich": (0.004, 0.45),      # 7 Tennessee farm watersheds (1940)
    "watt_chow": (0.01, 5840.0),   # 44 Canadian watersheds (1985)
    "giandotti": (10.0, 1000.0),   # Italian basins (1934)
}
DEFAULT_METHOD = "watt_chow"


@dataclass
class TimingResult:
    t_c_min: np.ndarray          # (B,) time of concentration, minutes
    t_peak_min: np.ndarray       # (B,) time to peak at the outlet, minutes
    within_fitted_range: np.ndarray   # (B,) bool
    channel_factor: float
    rain_duration_min: float
    method: str = DEFAULT_METHOD

    def report(self) -> str:
        n = self.t_c_min.size
        lo, hi = FITTED_RANGE_KM2[self.method]
        out = int(np.sum(~self.within_fitted_range))
        fin = np.isfinite(self.t_peak_min)
        lines = [f"time to peak ({self.method}): median "
                 f"{np.nanmedian(self.t_peak_min[fin]) if fin.any() else float('nan'):.0f} min "
                 f"over {int(fin.sum())}/{n} basins"]
        if out:
            lines.append(
                f"!!  {out}/{n} basins ({100*out/n:.0f}%) are outside "
                f"{self.method}'s fitted range ({lo}-{hi} km^2). These are "
                f"screening estimates, not defensible lead times.")
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


def _kirpich(L, S, **kw):
    with np.errstate(invalid="ignore", divide="ignore"):
        return KIRPICH_C_METRIC * np.power(L, 0.77) * np.power(S, -0.385)


def _watt_chow(L, S, **kw):
    """Watt & Chow (1985): t_c [h] = 0.000326 (L / sqrt(S))^0.79, L in m.

    Fitted on 44 watersheds spanning 0.01 to 5840 km^2, which BRACKETS our
    sub-basins instead of sitting two orders of magnitude below them. This
    is the default for that reason alone: it is the same kind of empirical
    regression as Kirpich, but the regression was actually run on catchments
    the size of ours.
    """
    with np.errstate(invalid="ignore", divide="ignore"):
        return 60.0 * 0.000326 * np.power(L / np.sqrt(S), 0.79)


def _giandotti(L, S, area_km2=None, mean_elev_above_outlet_m=None, **kw):
    """Giandotti (1934): t_c [h] = (4 sqrt(A) + 1.5 L_km) / (0.8 sqrt(Hm)).

    Designed for 10-1000 km^2 basins. Uses basin RELIEF rather than channel
    slope, so it fails differently from the other two -- which is the point
    of having it in the ensemble.
    """
    if area_km2 is None or mean_elev_above_outlet_m is None:
        return np.full(np.shape(L), np.nan)
    A = np.asarray(area_km2, dtype=np.float64)
    Hm = np.asarray(mean_elev_above_outlet_m, dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        t = (4.0 * np.sqrt(A) + 1.5 * (L / 1000.0)) / (0.8 * np.sqrt(Hm))
    return 60.0 * np.where(Hm > 0, t, np.nan)


METHODS = {"kirpich": _kirpich, "watt_chow": _watt_chow, "giandotti": _giandotti}


def time_of_concentration(length_m, slope, area_km2=None,
                          channel: str = "natural",
                          rain_duration_min: float = 0.0,
                          method: str = DEFAULT_METHOD,
                          mean_elev_above_outlet_m=None) -> TimingResult:
    """Kirpich t_c in minutes, with an out-of-range flag per basin.

    `t_peak_min` is filled here rather than left to the caller: an earlier
    version stored t_c in that field, and `report()` then printed t_c under
    the label "time to peak" -- 136 min where the true figure was 111. A
    quantity in a field named after a different quantity is the plainest
    possible way to publish a wrong warning time.
    """
    if channel not in CHANNEL_FACTORS:
        raise ValueError(f"channel must be one of {sorted(CHANNEL_FACTORS)}")
    if method not in METHODS:
        raise ValueError(f"method must be one of {sorted(METHODS)}")
    k = CHANNEL_FACTORS[channel]
    L = np.asarray(length_m, dtype=np.float64)
    S = np.asarray(slope, dtype=np.float64)
    tc = k * METHODS[method](L, S, area_km2=area_km2,
                             mean_elev_above_outlet_m=mean_elev_above_outlet_m)
    tc = np.asarray(tc, dtype=np.float64)
    needs_slope = method in ("kirpich", "watt_chow")
    good = (L > 0) & (np.isfinite(S) & (S > 0) if needs_slope else True)
    tc = np.where(good, tc, np.nan)

    lo, hi = FITTED_RANGE_KM2[method]
    if area_km2 is None:
        ok = np.ones(tc.shape, dtype=bool)
    else:
        a = np.asarray(area_km2, dtype=np.float64)
        ok = (a >= lo) & (a <= hi)
    return TimingResult(tc, time_to_peak(tc, rain_duration_min), ok, k,
                        float(rain_duration_min), method)


def compare_methods(length_m, slope, area_km2=None,
                    mean_elev_above_outlet_m=None,
                    rain_duration_min: float = 0.0,
                    channel: str = "natural") -> dict:
    """Every method side by side, plus the spread between them.

    The spread is the honest uncertainty on a warning lead time. Three
    independent empirical regressions disagreeing by a factor of two is
    information the operator needs; publishing whichever one was coded first,
    to the minute, is not.
    """
    out = {m: time_of_concentration(length_m, slope, area_km2, channel,
                                    rain_duration_min, m,
                                    mean_elev_above_outlet_m)
           for m in METHODS}
    stack = np.stack([np.asarray(v.t_peak_min, dtype=np.float64)
                      for v in out.values()])
    with np.errstate(invalid="ignore"):
        lo = np.nanmin(stack, axis=0)
        hi = np.nanmax(stack, axis=0)
        ratio = np.where(lo > 0, hi / lo, np.nan)
    return {"methods": out, "t_peak_min_low": lo, "t_peak_min_high": hi,
            "spread_ratio": ratio}


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
