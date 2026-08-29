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
                  max_offset: int = 12, grid_km: float = 4.0) -> SceneOffset:
    """One coincident INSAT/IMERG pair, already on the common grid."""
    r = alignment_offset(tir_k, rain_mmhr, tir_cold_k=tir_cold_k,
                         rain_mm=rain_mm, max_offset=max_offset, grid_km=grid_km)
    cold = np.isfinite(tir_k) & (np.asarray(tir_k) <= tir_cold_k)
    wet = np.isfinite(rain_mmhr) & (np.asarray(rain_mmhr) >= rain_mm)
    tz = np.tan(np.deg2rad(satellite_zenith(lat, lon)))
    return SceneOffset(scene=scene or "scene", dy=r.dy, dx=r.dx,
                       peak=r.peak_score, zero=r.zero_score,
                       n_cold=int(cold.sum()), n_wet=int(wet.sum()),
                       mean_tan_zenith=float(np.nanmean(np.where(cold, tz, np.nan)))
                       if cold.any() else float("nan"))


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
