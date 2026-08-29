"""SCS Curve Number runoff. Rain depth in, runoff depth out.

    S = 25400/CN - 254            (mm, metric form)
    Q = (P - 0.2S)^2 / (P + 0.8S) for P > 0.2S, else 0

THE TRAP IN THAT SECOND LINE
----------------------------
`(P - 0.2S)**2` is positive whether or not P exceeds 0.2S, so evaluating it
unconditionally produces runoff from rainfall that physically produces none
-- and it produces MORE of it the drier the storm, because the square grows
as P falls further below the initial abstraction. A 1 mm shower on dry
forest (CN 36, S = 452 mm, Ia = 90 mm) would report

    (1 - 90)^2 / (1 + 361) = 21.9 mm of runoff

from 1 mm of rain. Plausible-looking, catastrophically wrong, and it is the
same shape as every other bug this project has hit: a formula that keeps
returning numbers outside the domain it was derived on. `runoff_depth`
therefore masks P <= Ia FIRST and is tested against that specific case.

Antecedent moisture
-------------------
CN tables are AMC II (average). Flash floods do not happen on average
antecedent conditions -- they happen when the ground is already wet, and the
difference is not small: CN 70 dry (AMC I) is 51, wet (AMC III) is 85, which
on 80 mm of rain is 5 mm of runoff versus 42 mm. An eightfold spread that
depends entirely on the previous five days. Modelling AMC II only would make
the flood head systematically under-forecast exactly the cases it exists for,
so `adjust_amc` is provided and `route()` requires the caller to state which
condition it is using rather than defaulting silently.

Sources: NRCS TR-55 (1986) Table 2-2; Chow, Maidment & Mays (1988) for the
AMC conversions.
"""
from __future__ import annotations

import numpy as np

# ESA WorldCover 2021 class -> CN for hydrologic soil groups A, B, C, D.
# (WorldCover is the recommendation in docs/FLOOD_DATA.md; the mapping is a
# dict so a Bhuvan LULC legend can be substituted without touching the math.)
WORLDCOVER_CN = {
    10:  (36, 60, 73, 79),   # tree cover        -- woods, fair condition
    20:  (35, 56, 70, 77),   # shrubland         -- brush, fair
    30:  (49, 69, 79, 84),   # grassland         -- pasture, fair
    40:  (67, 78, 85, 89),   # cropland          -- row crops, straight, good
    50:  (77, 85, 90, 92),   # built-up          -- residential, ~65% impervious
    60:  (77, 86, 91, 94),   # bare / sparse     -- fallow, bare soil
    70:  (98, 98, 98, 98),   # snow and ice      -- see SNOW_IS_NOT_MODELLED
    80:  (100, 100, 100, 100),  # permanent water -- rain on water is runoff
    90:  (85, 85, 85, 85),   # herbaceous wetland -- saturated, group-invariant
    95:  (85, 85, 85, 85),   # mangroves
    100: (30, 58, 71, 78),   # moss and lichen   -- tundra
}

# Snow and ice get an impervious-like CN because this is a RAINFALL-runoff
# model with no snowmelt term. In the Himalaya that is wrong in both
# directions -- rain-on-snow enhances runoff, seasonal storage delays it --
# so basins dominated by class 70 are flagged rather than quietly scored.
SNOW_IS_NOT_MODELLED = 70

HSG_NAMES = ("A", "B", "C", "D")     # HYSOGs250m codes 1..4 (also 11..14 = drained)


def curve_number(landcover: np.ndarray, hsg: np.ndarray,
                 table: dict | None = None,
                 default: float = np.nan) -> np.ndarray:
    """CN(II) from a land-cover raster and a hydrologic-soil-group raster.

    hsg is 1..4 for A..D. HYSOGs250m also uses 11..14 for "drained" versions
    of A..D; those are mapped to their undrained group rather than dropped,
    which is conservative (undrained runs off more) and stated here because
    silently discarding them would blank large parts of the Indo-Gangetic
    plain.

    Unmapped combinations return `default` (nan) rather than a guess.
    """
    table = table or WORLDCOVER_CN
    lc = np.asarray(landcover)
    sg = np.asarray(hsg)
    if lc.shape != sg.shape:
        raise ValueError(f"shape mismatch: landcover {lc.shape} vs hsg {sg.shape}")

    grp = np.where(sg >= 11, sg - 10, sg)            # drained -> parent group
    out = np.full(lc.shape, default, dtype=np.float64)
    valid_grp = (grp >= 1) & (grp <= 4)
    for cls, cns in table.items():
        m = (lc == cls) & valid_grp
        if not m.any():
            continue
        idx = grp[m].astype(int) - 1
        out[m] = np.asarray(cns, dtype=np.float64)[idx]
    return out


def adjust_amc(cn2: np.ndarray, condition: str) -> np.ndarray:
    """AMC II -> I (dry) or III (wet). Chow et al. (1988).

        CN_I   = 4.2 CN / (10 - 0.058 CN)
        CN_III =  23 CN / (10 + 0.13  CN)
    """
    cn = np.asarray(cn2, dtype=np.float64)
    if condition == "II":
        return cn
    if condition == "I":
        return 4.2 * cn / (10.0 - 0.058 * cn)
    if condition == "III":
        return 23.0 * cn / (10.0 + 0.13 * cn)
    raise ValueError(f"condition must be 'I', 'II' or 'III', got {condition!r}")


def potential_retention(cn: np.ndarray) -> np.ndarray:
    """S in mm. CN <= 0 is undefined, not infinite -- returns nan."""
    cn = np.asarray(cn, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        s = 25400.0 / cn - 254.0
    return np.where((cn > 0) & (cn <= 100), s, np.nan)


def runoff_depth(rain_mm: np.ndarray, cn: np.ndarray,
                 ia_ratio: float = 0.2) -> np.ndarray:
    """Direct runoff depth Q (mm) from rainfall depth P (mm).

    Zero where P <= Ia = ia_ratio * S. See the module docstring: evaluating
    the quotient below that threshold invents runoff, and invents more of it
    the smaller the storm.

    `ia_ratio` is exposed because 0.2 is a 1950s US calibration that many
    later studies put nearer 0.05 for other regions; it is a judgment call,
    so it is a parameter and is recorded in the result rather than buried.
    """
    p = np.asarray(rain_mm, dtype=np.float64)
    s = potential_retention(cn)
    ia = ia_ratio * s
    with np.errstate(invalid="ignore", divide="ignore"):
        q = (p - ia) ** 2 / (p - ia + s)
    q = np.where(np.isfinite(s) & (p > ia), q, 0.0)
    return np.where(np.isfinite(s) & np.isfinite(p), q, np.nan)
