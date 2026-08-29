"""Antecedent moisture from IMERG. The largest single term in the flood chain.

Measured on our own numbers: 80 mm of rain on CN 70 produces 2.8 mm of
runoff under dry antecedent conditions and 42.2 mm under wet -- a factor of
**15.3** from the previous five days alone. That is larger than the spread
the rainfall model itself is likely to produce, so a fixed AMC assumption
does not make the flood head approximately right; it makes it wrong in one
direction roughly half the time, and wrong in the direction that matters
(under-forecasting) on exactly the saturated-catchment days flash floods
actually occur.

So AMC is an INPUT, derived from the 5-day antecedent IMERG accumulation.
That makes it another channel in the frozen cache config, and it has to be
there before bulk ingest -- adding it later changes the cache fingerprint
and forces a re-cache.

Thresholds are the NRCS ones (TR-55 / NEH-4), 5-day totals:

    season     AMC I (dry)     AMC II          AMC III (wet)
    growing    < 35.6 mm       35.6 - 53.3     > 53.3 mm
    dormant    < 12.7 mm       12.7 - 27.9     > 27.9 mm

The Indian monsoon is a growing season throughout; `season="growing"` is
the default and the dormant thresholds matter only for winter western
disturbances.

WHY THE DEFAULT IS CONTINUOUS, NOT THREE CLASSES
------------------------------------------------
The NRCS classes are a step function: 35.5 mm of antecedent rain gives CN
49.5, and 35.7 mm gives CN 70. For an operational product that is a
discontinuity in published risk driven by 0.2 mm of rain five days ago --
two neighbouring basins either side of the threshold get visibly different
warnings for no physical reason, and the same basin's risk jumps between
consecutive runs. `curve_number_from_antecedent` interpolates through the
class anchors instead, reproducing the standard values exactly at the
boundaries while removing the cliff between them.
"""
from __future__ import annotations

import numpy as np

from .curve_number import adjust_amc

# (dry_max_mm, wet_min_mm) 5-day antecedent totals
AMC_THRESHOLDS = {"growing": (35.6, 53.3), "dormant": (12.7, 27.9)}


def antecedent_5day(daily_mm: np.ndarray, axis: int = 0) -> np.ndarray:
    """Sum the 5 days BEFORE each day. Excludes the day itself.

    Including the current day would leak the event being forecast into its
    own antecedent condition -- a storm would raise its own CN and inflate
    its own runoff. That is target leakage in a physics term, and it would
    improve every hindcast score while being unavailable at forecast time.
    """
    a = np.asarray(daily_mm, dtype=np.float64)
    a = np.moveaxis(a, axis, 0)
    if a.shape[0] < 6:
        raise ValueError(f"need at least 6 days to form a 5-day antecedent "
                         f"total for one day, got {a.shape[0]}")
    csum = np.cumsum(np.nan_to_num(a), axis=0)
    out = np.full_like(a, np.nan)
    out[5:] = csum[4:-1] - np.concatenate(
        [np.zeros((1,) + a.shape[1:]), csum[:-6]], axis=0)
    return np.moveaxis(out, 0, axis)


def amc_class(p5_mm, season: str = "growing") -> np.ndarray:
    """NRCS class 1 (dry), 2 (average) or 3 (wet). 0 where undefined."""
    if season not in AMC_THRESHOLDS:
        raise ValueError(f"season must be one of {sorted(AMC_THRESHOLDS)}")
    lo, hi = AMC_THRESHOLDS[season]
    p = np.asarray(p5_mm, dtype=np.float64)
    cls = np.where(p < lo, 1, np.where(p > hi, 3, 2))
    return np.where(np.isfinite(p), cls, 0).astype(np.int8)


def curve_number_from_antecedent(cn2, p5_mm, season: str = "growing",
                                 continuous: bool = True) -> np.ndarray:
    """CN(II) + 5-day antecedent rain -> the CN to actually use.

    continuous=True (default) interpolates piecewise-linearly through the
    three NRCS anchors -- CN_I at the dry boundary, CN_II at the midpoint of
    the average band, CN_III at the wet boundary -- so the standard values
    are reproduced exactly where the classes are defined and there is no
    cliff between them. See the module docstring.
    """
    if season not in AMC_THRESHOLDS:
        raise ValueError(f"season must be one of {sorted(AMC_THRESHOLDS)}")
    lo, hi = AMC_THRESHOLDS[season]
    cn2 = np.asarray(cn2, dtype=np.float64)
    p = np.asarray(p5_mm, dtype=np.float64)

    cn1 = adjust_amc(cn2, "I")
    cn3 = adjust_amc(cn2, "III")
    if not continuous:
        cls = amc_class(p, season)
        out = np.where(cls == 1, cn1, np.where(cls == 3, cn3, cn2))
        return np.where(cls == 0, np.nan, out)

    mid = 0.5 * (lo + hi)
    out = np.interp(p, [0.0, lo, mid, hi], [0.0, 1.0, 2.0, 3.0])   # anchor axis
    # piecewise-linear in CN between the anchors
    below = np.clip((out - 1.0), 0.0, 1.0)      # CN_I  -> CN_II
    above = np.clip((out - 2.0), 0.0, 1.0)      # CN_II -> CN_III
    cn = np.where(out <= 2.0, cn1 + below * (cn2 - cn1),
                  cn2 + above * (cn3 - cn2))
    return np.where(np.isfinite(p) & np.isfinite(cn2), cn, np.nan)
