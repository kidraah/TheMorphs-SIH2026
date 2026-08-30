"""The alignment gate: does INSAT cold cloud sit where IMERG shows rain?

This is the check that clears all Indian training data, and it is the failure
mode with no downstream symptom. A systematic offset does not raise, does not
degrade the loss, and does not show up in any score: the model simply learns
"rain appears N km from the cold top", a stable and completely wrong
relationship, and the warnings come out in the wrong valley.

Strict, but strict about the RIGHT quantity
-------------------------------------------
A naive "offset must be under one pixel" gate would FAIL correct data. A
cloud TOP is not above the rain it produces: it is displaced away from the
sub-satellite point by h*tan(satellite zenith angle). For INSAT-3DR at 74E
and a 12 km top, that is

    Kanyakumari  0.5 px      Hyderabad  1.2 px
    Delhi        2.0 px      Kashmir    2.5 px

Real, physical, and largest over precisely the cloudburst region. So the gate
decomposes the measured offset into

    constant component  -- independent of viewing geometry -> GEOREFERENCING,
                           the bug we are hunting. Strict threshold.
    parallax component  -- scaling with tan(zenith) -> physics. Reported, and
                           NOT a failure.

A gate that failed on parallax would be ignored within a week, which is worse
than no gate.

What it reports
---------------
    per-scene offset          (dy, dx) in pixels and km, per scene pair
    latitude-banded offset    to expose the parallax signature
    constant component        the georeferencing error estimate
    parallax slope            km of displacement per unit tan(zenith)
    identifiability           whether the peak is sharp enough to trust
    consistency               spread of the offset across scenes
    VERDICT                   PASS / FAIL / INCONCLUSIVE, with reasons
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .alignment import alignment_offset

R_EARTH_KM = 6371.0
H_GEO_KM = 35786.0

# --- pass thresholds -------------------------------------------------------
# The georeferencing (constant) component. One 4 km cell: anything larger is
# a systematic shift we should explain before training on it.
MAX_CONSTANT_OFFSET_PX = 1.0
# Spread of per-scene offsets. A real georeferencing error is consistent; a
# large spread means we are measuring storm motion, not registration.
MAX_OFFSET_IQR_PX = 2.0
# The peak must beat zero-offset agreement by this factor to be identifiable.
# Below it, the surface is flat and the "offset" is noise.
MIN_PEAK_RATIO = 1.15
# Minimum evidence per scene, and scenes overall.
MIN_COLD_PIXELS = 1000
MIN_WET_PIXELS = 1000
MIN_SCENES = 5

# Cut each scene into zenith bands and regress on the bands, not on
# scene-mean tan(zenith).
#
# MEASURED over the India grid from 74E: tan(zenith) runs 0.130 to 1.322,
# within-scene sd 0.247. Across the 21 measured scenes, the sd of the
# SCENE-MEAN tan(zenith) is 0.0413 -- six times smaller. Regression precision
# on the slope goes as 1/(sd_x * sqrt(N)), and banding raises sd_x by 6x while
# the extra per-band noise (fewer pixels) cancels against the extra points.
#
# Simulated recovery of a true 3.0 px (12 km) slope, 0.5 px band noise,
# 90% interval:
#
#     scenes   between-scene        within-scene bands
#          3   [-19.75, +27.80]     [+0.94, +5.01]
#          6   [ -8.04, +13.80]     [+1.58, +4.41]
#         21   [ -1.70,  +7.56]     [+2.27, +3.73]
#
# The between-scene design does not resolve 12 km at ANY scene count -- its
# interval still spans zero at 21 scenes. That is a design limit, not a data
# shortage, and no amount of downloading fixes it.
DEFAULT_TILE_PX = 128        # 512 km at 4 km/px


def satellite_zenith(lat, lon, sub_lon: float = 74.0) -> np.ndarray:
    """Satellite zenith angle in degrees for a geostationary sub-longitude."""
    la = np.deg2rad(np.asarray(lat, dtype=np.float64))
    dlo = np.deg2rad(np.asarray(lon, dtype=np.float64) - sub_lon)
    psi = np.arccos(np.clip(np.cos(la) * np.cos(dlo), -1.0, 1.0))
    return np.degrees(np.arctan2((R_EARTH_KM + H_GEO_KM) * np.sin(psi),
                                 (R_EARTH_KM + H_GEO_KM) * np.cos(psi) - R_EARTH_KM))


def expected_parallax_km(lat, lon, cloud_top_km: float = 12.0,
                         sub_lon: float = 74.0) -> np.ndarray:
    """Displacement of a cloud top from the ground point beneath it."""
    return cloud_top_km * np.tan(np.deg2rad(satellite_zenith(lat, lon, sub_lon)))


@dataclass
class SceneOffset:
    scene: str
    dy: int
    dx: int
    peak: float
    zero: float
    n_cold: int
    n_wet: int
    mean_tan_zenith: float
    sub_lon: float = 74.0

    @property
    def magnitude_px(self) -> float:
        return float(np.hypot(self.dy, self.dx))

    @property
    def identifiable(self) -> bool:
        return (self.zero > 0 and self.peak / max(self.zero, 1e-9) >= MIN_PEAK_RATIO
                and self.n_cold >= MIN_COLD_PIXELS and self.n_wet >= MIN_WET_PIXELS)


@dataclass
class GateResult:
    scenes: list = field(default_factory=list)
    grid_km: float = 4.0
    constant_px: tuple = (float("nan"), float("nan"))
    parallax_slope_km: float = float("nan")
    reasons: list = field(default_factory=list)
    verdict: str = "INCONCLUSIVE"
    # Cluster-robust extras, populated by run_gate_tiled.
    scene_level_sd_px: float = float("nan")
    constant_se_px: float = float("nan")
    n_scenes_used: int = 0
    n_tiles_used: int = 0

    @property
    def usable(self) -> list:
        return [s for s in self.scenes if s.identifiable]

    def report(self) -> str:
        L = ["=" * 72, f"ALIGNMENT GATE: {self.verdict}", "=" * 72, ""]
        L.append(f"{'scene':>22} {'dy':>4} {'dx':>4} {'|off| km':>9} "
                 f"{'peak':>7} {'zero':>7} {'cold':>8} {'wet':>8} {'ident':>6}")
        L.append("-" * 88)
        for s in self.scenes:
            L.append(f"{s.scene[:22]:>22} {s.dy:>4d} {s.dx:>4d} "
                     f"{s.magnitude_px * self.grid_km:>9.1f} {s.peak:>7.4f} "
                     f"{s.zero:>7.4f} {s.n_cold:>8,} {s.n_wet:>8,} "
                     f"{'yes' if s.identifiable else 'NO':>6}")
        L.append("")
        cy, cx = self.constant_px
        if np.isfinite(cy):
            L.append(f"constant (georeferencing) component: "
                     f"dy {cy:+.2f} px, dx {cx:+.2f} px "
                     f"= ({cy * self.grid_km:+.1f}, {cx * self.grid_km:+.1f}) km")
            L.append(f"   threshold: |constant| <= {MAX_CONSTANT_OFFSET_PX:.1f} px "
                     f"({MAX_CONSTANT_OFFSET_PX * self.grid_km:.0f} km)")
        if np.isfinite(self.parallax_slope_km):
            L.append(f"parallax component: {self.parallax_slope_km:+.1f} km per unit "
                     f"tan(zenith)  (a 12 km cloud top predicts ~12)")
        L.append("")
        for r in self.reasons:
            L.append(f"  {r}")
        if self.verdict == "FAIL":
            L.append("")
            L.append("  DO NOT BUILD TRAINING DATA. A systematic offset has no")
            L.append("  downstream symptom: training proceeds, the loss falls, and the")
            L.append("  model learns a stable wrong cloud-to-rain displacement while")
            L.append("  every score looks healthy.")
        return "\n".join(L)


def measure_scene(tir_k, rain_mmhr, lat, lon, scene: str = "",
                  tir_cold_k: float = 240.0, rain_mm: float = 0.5,
                  max_offset: int = 12, grid_km: float = 4.0,
                  sub_lon: float = 74.0) -> SceneOffset:
    """One coincident INSAT/IMERG pair, already on the common grid.

    `sub_lon` is the SATELLITE's sub-point longitude and must be passed per
    scene: INSAT-3DR sits at 74E and 3DS at 82E, so the same ground cell has
    a different zenith angle depending on which satellite saw it. Defaulting
    it silently would mix two geometries into one regression and corrupt the
    parallax slope -- read it from the file's
    Nominal_Central_Point_Coordinates attribute rather than assuming.
    """
    r = alignment_offset(tir_k, rain_mmhr, tir_cold_k=tir_cold_k,
                         rain_mm=rain_mm, max_offset=max_offset, grid_km=grid_km)
    cold = np.isfinite(tir_k) & (np.asarray(tir_k) <= tir_cold_k)
    wet = np.isfinite(rain_mmhr) & (np.asarray(rain_mmhr) >= rain_mm)
    tz = np.tan(np.deg2rad(satellite_zenith(lat, lon, sub_lon=sub_lon)))
    return SceneOffset(scene=scene or "scene", dy=r.dy, dx=r.dx,
                       peak=r.peak_score, zero=r.zero_score,
                       n_cold=int(cold.sum()), n_wet=int(wet.sum()),
                       mean_tan_zenith=float(np.nanmean(np.where(cold, tz, np.nan)))
                       if cold.any() else float("nan"),
                       sub_lon=float(sub_lon))


def measure_scene_tiled(tir_k, rain_mmhr, lat, lon, scene: str = "",
                        tile_px: int = DEFAULT_TILE_PX, sub_lon: float = 74.0,
                        min_cold: int = MIN_COLD_PIXELS, **kw) -> list:
    """One scene -> one SceneOffset per COMPACT tile.

    Tiles, not zenith bands. The first version cut each scene into quantile
    bands of tan(zenith), which gives a wide lever arm and does not work:
    a band is a thin annulus, so the displacement component ALONG the strip
    is unconstrained and the argmax runs to the search boundary. Measured on
    a synthetic with a known (-5, -5) displacement, every band recovered one
    axis and saturated the other at -12.

    A compact square tile constrains both axes, and tiles still span most of
    the domain's tan(zenith) range because that varies with position. Only
    the reference (cloud top) is masked; the rain field stays whole, or a
    shift would move it off the tile and zero offset would always win.
    """
    tir = np.asarray(tir_k, dtype=np.float64)
    rain = np.asarray(rain_mmhr, dtype=np.float64)
    h, w = tir.shape
    out = []
    for r0 in range(0, h - tile_px + 1, tile_px):
        for c0 in range(0, w - tile_px + 1, tile_px):
            m = np.zeros_like(tir, dtype=bool)
            m[r0:r0 + tile_px, c0:c0 + tile_px] = True
            cold = m & np.isfinite(tir) & (tir <= kw.get("tir_cold_k", 240.0))
            if int(cold.sum()) < min_cold:
                continue          # no convection in this tile; not a failure
            try:
                out.append(measure_scene(
                    np.where(m, tir, np.nan), rain, lat, lon,
                    scene=f"{scene}/r{r0//tile_px}c{c0//tile_px}",
                    sub_lon=sub_lon, **kw))
            except ValueError:
                continue
    return out


def run_gate(scenes: list, grid_km: float = 4.0) -> GateResult:
    """Aggregate per-scene offsets into a verdict.

    `scenes` is a list of SceneOffset. The decomposition regresses the
    meridional offset on tan(zenith): the intercept is the georeferencing
    error, the slope is parallax.
    """
    res = GateResult(scenes=list(scenes), grid_km=grid_km)
    usable = res.usable

    if len(res.scenes) < MIN_SCENES:
        res.verdict = "INCONCLUSIVE"
        res.reasons.append(
            f"only {len(res.scenes)} scene(s), need >= {MIN_SCENES}. One pair "
            f"cannot separate a registration offset from storm motion.")
        return res
    if len(usable) < MIN_SCENES:
        res.verdict = "INCONCLUSIVE"
        res.reasons.append(
            f"only {len(usable)} of {len(res.scenes)} scenes are identifiable "
            f"(peak must beat zero-offset by {MIN_PEAK_RATIO:.2f}x with "
            f">={MIN_COLD_PIXELS:,} cold and >={MIN_WET_PIXELS:,} wet pixels). "
            f"A flat agreement surface means there is no convection to align.")
        return res

    dy = np.array([s.dy for s in usable], dtype=float)
    dx = np.array([s.dx for s in usable], dtype=float)
    tz = np.array([s.mean_tan_zenith for s in usable], dtype=float)

    # Meridional offset vs tan(zenith): intercept = georeferencing, slope =
    # parallax. Fall back to the median when the geometry does not vary enough
    # across scenes to identify a slope.
    if np.isfinite(tz).all() and np.ptp(tz) > 0.05:
        slope, intercept = np.polyfit(tz, dy, 1)
        res.parallax_slope_km = float(slope * grid_km)
        const_y = float(intercept)
    else:
        res.parallax_slope_km = float("nan")
        const_y = float(np.median(dy))
        res.reasons.append(
            "viewing geometry barely varies across these scenes, so parallax "
            "and a constant offset cannot be separated; using the median.")
    const_x = float(np.median(dx))
    res.constant_px = (const_y, const_x)

    iqr = max(float(np.subtract(*np.percentile(dy, [75, 25]))),
              float(np.subtract(*np.percentile(dx, [75, 25]))))

    ok = True
    mag = float(np.hypot(const_y, const_x))
    if mag > MAX_CONSTANT_OFFSET_PX:
        ok = False
        res.reasons.append(
            f"FAIL: constant offset {mag:.2f} px ({mag * grid_km:.1f} km) exceeds "
            f"{MAX_CONSTANT_OFFSET_PX:.1f} px. This is independent of viewing "
            f"geometry, so it is NOT parallax -- it is a georeferencing error.")
    else:
        res.reasons.append(
            f"constant offset {mag:.2f} px ({mag * grid_km:.1f} km) is within "
            f"{MAX_CONSTANT_OFFSET_PX:.1f} px.")

    if iqr > MAX_OFFSET_IQR_PX:
        ok = False
        res.reasons.append(
            f"FAIL: per-scene offsets spread by {iqr:.2f} px (IQR), above "
            f"{MAX_OFFSET_IQR_PX:.1f}. A registration error is consistent; this "
            f"much scatter means storm motion is being measured instead.")
    else:
        res.reasons.append(f"per-scene spread {iqr:.2f} px (IQR) is consistent.")

    if np.isfinite(res.parallax_slope_km):
        res.reasons.append(
            f"parallax slope {res.parallax_slope_km:+.1f} km per unit tan(zenith); "
            f"a 12 km cloud top predicts ~12, and this component is EXPECTED, "
            f"not a defect.")

    res.verdict = "PASS" if ok else "FAIL"
    return res


# ---------------------------------------------------------------------------
# Scene selection -- run BEFORE the gate, not after
# ---------------------------------------------------------------------------
#
# The gate returns INCONCLUSIVE on a day with no convection, which is correct
# but wastes a cycle. Worse, a day with plenty of BROAD rain also fails to
# identify an offset: stratiform shields overlap at any small shift, so the
# agreement surface is flat. Measured on 2025-08-01 2345Z -- 87k cold cells,
# 72k wet cells, and a peak/zero ratio of only 1.04.
#
# What identifies an offset is CELLULAR convection: many separate heavy cores,
# each a sharp feature whose displacement is unambiguous. So candidate scenes
# are scored on core count x heavy fraction, and the best are chosen before
# any INSAT scan is downloaded.

CELLULARITY_HEAVY_MM_HR = 5.0


def scene_cellularity(rain_mmhr, lats=None, lons=None, box=None) -> dict:
    """Score a candidate IMERG granule for alignment usefulness.

    Returns wet/heavy fractions, the number of separate heavy cores, and a
    combined score. Higher is better: many distinct cores give a sharp peak.
    """
    from scipy import ndimage

    r = np.asarray(rain_mmhr, dtype=np.float64)
    if lats is not None and lons is not None and box is not None:
        la = (lats >= box[1]) & (lats <= box[3])
        lo = (lons >= box[0]) & (lons <= box[2])
        r = r[np.ix_(la, lo)]

    v = r[np.isfinite(r)]
    if v.size == 0:
        return dict(wet_pct=0.0, heavy_pct=0.0, cores=0, max_mm_hr=0.0, score=0.0)

    heavy = np.nan_to_num(r) >= CELLULARITY_HEAVY_MM_HR
    lab, n = ndimage.label(heavy)
    heavy_pct = 100.0 * float((v >= CELLULARITY_HEAVY_MM_HR).mean())
    return dict(wet_pct=100.0 * float((v >= 0.5).mean()),
                heavy_pct=heavy_pct, cores=int(n), max_mm_hr=float(v.max()),
                score=float(n) * heavy_pct)


def rank_candidate_scenes(scored: dict, n: int = 6) -> list:
    """Best n candidates by cellularity. `scored` maps label -> scene_cellularity()."""
    return sorted(scored.items(), key=lambda kv: -kv[1]["score"])[:n]


# ---------------------------------------------------------------------------
# Pooling: 21 weak measurements are not 21 argmaxes
# ---------------------------------------------------------------------------
#
# Measured on 2025-08-01: 21 scenes, every one with >70,000 cold and >45,000
# wet pixels, and NOT ONE identifiable -- peak/zero median 1.017 against a
# 1.15 bar. Monsoon stratiform overlaps at any small shift, so each scene's
# agreement surface is flat, exactly as this module's scene-selection note
# predicted.
#
# The argmaxes nonetheless cluster: mean |dy| = 1.62 where a flat surface
# searched over +/-12 px would give ~6. That is suggestive and it is NOT a
# result -- reading direction off surfaces the gate calls unidentifiable is
# how this project previously talked itself into an anvil-advection story
# that ERA5 then refuted.
#
# The right response is a better statistic, not a looser threshold. A
# constant registration offset is COMMON to every scene, so the surfaces can
# be summed: 21 x ~100k cold pixels is ~2M, and a real common offset
# reinforces while storm-motion scatter does not. Each surface is divided by
# its own zero-offset score first, because scenes differ in absolute IoU
# (0.21 to 0.44 here) and summing raw would just weight the wettest scenes.
#
# WHAT POOLING CAN AND CANNOT RECOVER. It recovers the CONSTANT component
# only. Parallax varies with tan(zenith), so pooling scenes of differing
# geometry smears it into the constant. Report it as a constant estimate and
# keep the regression for the parallax term.

def pooled_offset(results, grid_km: float = 4.0) -> dict:
    """Sum per-scene agreement surfaces and locate the common offset."""
    surfaces = [np.asarray(r.scores, dtype=np.float64) for r in results]
    if not surfaces:
        return {"identifiable": False, "reason": "no surfaces"}
    if len({s.shape for s in surfaces}) != 1:
        raise ValueError("surfaces must share a shape (same max_offset)")

    n = surfaces[0].shape[0]
    c = n // 2
    norm = []
    for s in surfaces:
        z = s[c, c]
        if z > 0:
            norm.append(s / z)
    if not norm:
        return {"identifiable": False, "reason": "every zero-offset score was 0"}

    pooled = np.mean(norm, axis=0)
    k = int(np.argmax(pooled))
    iy, ix = divmod(k, n)
    peak, zero = float(pooled[iy, ix]), float(pooled[c, c])
    ratio = peak / zero if zero > 0 else float("nan")

    # A pooled surface is smoother, so the bar should not be the per-scene
    # one. What matters is whether the peak stands above the surface's own
    # roughness: compare the improvement over zero with the spread of the
    # pooled surface away from the peak.
    others = np.delete(pooled.ravel(), k)
    z_score = (peak - others.mean()) / (others.std() or 1e-12)

    return {
        "dy": iy - c, "dx": ix - c,
        "offset_km": float(np.hypot(iy - c, ix - c) * grid_km),
        "peak": peak, "zero": zero, "peak_over_zero": ratio,
        "z_score": float(z_score),
        "n_scenes": len(norm),
        "identifiable": bool(z_score >= 3.0),
        "reason": ("pooled peak stands %.1f sigma above the surface" % z_score
                   if z_score >= 3.0 else
                   "pooled peak is only %.1f sigma above the surface -- still "
                   "not a measurement" % z_score),
    }


# ---------------------------------------------------------------------------
# Tiles within a scene are NOT independent
# ---------------------------------------------------------------------------
#
# Tiles cut from one scene share its storm field, its scan time and its
# registration:
#
#     dy_tile = const + slope * tz_tile + u_scene + e_tile
#
# u_scene is one draw per scene -- advection over the scan/IMERG time
# mismatch, that scan's own registration error, that half-hour's IMERG
# retrieval bias. Pooling tiles as if they were independent is the
# day-blocking and basin-weighting mistake one level down, and it makes the
# intervals too narrow.
#
# The two parameters are affected COMPLETELY DIFFERENTLY, which is what makes
# this worth handling rather than merely warning about. Simulated at
# sd(u) = 0.6 px, 3 scenes, 95% error:
#
#     tiles/scene      slope     constant
#               6       1.50         1.21
#              24       0.69         0.82
#              96       0.33         0.68
#             384       0.16         0.69      <- flat
#
# The slope keeps improving with tiles, because it is identified from
# WITHIN-scene variation in tan(zenith) and a common shift is absorbed by the
# intercept. The constant flattens at the scene-level floor, Var(u)/n_scenes,
# which no amount of tiling reduces.
#
#     TILES BUY THE SLOPE. ONLY SCENES BUY THE CONSTANT.
#
# So the slope is fit on pooled tiles, and the constant is formed from
# per-scene estimates with a between-scene standard error -- cluster-robust,
# clustered on the scene.

def run_gate_tiled(scene_tiles: dict, grid_km: float = 4.0) -> GateResult:
    """Gate on tiled measurements, clustering on the scene.

    `scene_tiles` maps a scene name to its list of per-tile SceneOffset.
    """
    usable = {k: [t for t in v if t.identifiable] for k, v in scene_tiles.items()}
    usable = {k: v for k, v in usable.items() if len(v) >= 3}
    flat = [t for v in usable.values() for t in v]
    res = GateResult(scenes=flat, grid_km=grid_km)

    if len(usable) < 2:
        res.verdict = "INCONCLUSIVE"
        res.reasons.append(
            f"{len(usable)} scene(s) with >=3 identifiable tiles. The constant "
            f"cannot be separated from a single scene's own registration and "
            f"advection -- that term is common to every tile of a scene, so "
            f"tiles do not help it. At least 2 scenes, preferably 3.")
        return res

    tz = np.array([t.mean_tan_zenith for t in flat], dtype=float)
    dy = np.array([t.dy for t in flat], dtype=float)
    dx = np.array([t.dx for t in flat], dtype=float)

    # Slope from pooled tiles: within-scene tan(zenith) variation identifies it.
    if np.isfinite(tz).all() and float(np.max(tz) - np.min(tz)) > 0.05:
        slope, _ = np.polyfit(tz, dy, 1)
        res.parallax_slope_km = float(slope * grid_km)
    else:
        slope = 0.0
        res.parallax_slope_km = float("nan")

    # Constant per SCENE, with the parallax term removed, then combined
    # across scenes. n here is the number of scenes, never the tile count.
    per_scene = []
    for name, tiles in usable.items():
        t_tz = np.array([t.mean_tan_zenith for t in tiles], dtype=float)
        t_dy = np.array([t.dy for t in tiles], dtype=float)
        per_scene.append(float(np.mean(t_dy - slope * t_tz)))
    per_scene = np.array(per_scene)
    const_y = float(np.mean(per_scene))
    n_sc = len(per_scene)
    scene_sd = float(np.std(per_scene, ddof=1)) if n_sc > 1 else float("nan")
    se_const = scene_sd / np.sqrt(n_sc) if np.isfinite(scene_sd) else float("nan")

    const_x = float(np.median([np.median([t.dx for t in v]) for v in usable.values()]))
    res.constant_px = (const_y, const_x)
    res.scene_level_sd_px = scene_sd
    res.constant_se_px = se_const
    res.n_scenes_used = n_sc
    res.n_tiles_used = len(flat)

    ok = True
    mag = float(np.hypot(const_y, const_x))
    res.reasons.append(
        f"{n_sc} scenes, {len(flat)} identifiable tiles. Constant estimated "
        f"from {n_sc} scene-level values (NOT {len(flat)} tiles): "
        f"{mag:.2f} px +/- {1.96*se_const:.2f} (95%).")
    res.reasons.append(
        f"scene-level scatter sd(u) = {scene_sd:.2f} px -- the floor on the "
        f"constant. Tiles cannot reduce it; only more scenes can.")
    if mag - 1.96 * se_const > MAX_CONSTANT_OFFSET_PX:
        ok = False
        res.reasons.append(
            f"FAIL: constant {mag:.2f} px exceeds {MAX_CONSTANT_OFFSET_PX:.1f} px "
            f"even at the lower end of its interval. Independent of viewing "
            f"geometry, so this is georeferencing, not parallax.")
    elif mag + 1.96 * se_const > MAX_CONSTANT_OFFSET_PX:
        res.verdict = "INCONCLUSIVE"
        res.reasons.append(
            f"the constant's 95% interval straddles the {MAX_CONSTANT_OFFSET_PX:.1f} px "
            f"bar ({mag - 1.96*se_const:.2f}..{mag + 1.96*se_const:.2f}). "
            f"Add scenes -- with sd(u) = {scene_sd:.2f}, "
            f"{int(np.ceil((1.96*scene_sd/max(MAX_CONSTANT_OFFSET_PX-mag,0.05))**2))} "
            f"scenes would resolve it.")
        return res
    else:
        res.reasons.append(
            f"constant {mag:.2f} px is within {MAX_CONSTANT_OFFSET_PX:.1f} px "
            f"across its whole interval.")

    if np.isfinite(res.parallax_slope_km):
        res.reasons.append(
            f"parallax slope {res.parallax_slope_km:+.1f} km per unit "
            f"tan(zenith); a 12 km cloud top predicts ~12. EXPECTED, not a defect.")
    res.verdict = "PASS" if ok else "FAIL"
    return res
