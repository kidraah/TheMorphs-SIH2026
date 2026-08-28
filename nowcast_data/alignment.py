"""Cross-check that INSAT and IMERG land in the same place on the grid.

The failure this catches
------------------------
A systematic geolocation offset between the inputs and the labels is the
worst bug in the pipeline, because nothing downstream reports it. Training
proceeds, the loss falls, and the model learns "rain appears ~30 km east of
the cold cloud top" -- a stable, learnable, completely wrong relationship.
Every score looks fine. It surfaces only when the system is deployed and the
warnings are in the wrong valley.

It is cheap to detect now and expensive to discover later, so it runs as a
gate before any Indian training data is built.

Method
------
Convective cells are cold in TIR-1 and raining in IMERG, so the two binary
fields should overlap. Slide one over the other across a range of pixel
offsets and find the offset of maximum agreement. A correctly aligned pair
peaks at (0, 0). A peak elsewhere is the offset, in 4 km pixels.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class AlignmentResult:
    dy: int                     # rows; positive = rain is SOUTH of cloud
    dx: int                     # cols; positive = rain is EAST of cloud
    peak_score: float
    zero_score: float
    grid_km: float
    scores: np.ndarray
    max_offset: int

    @property
    def offset_km(self) -> tuple[float, float]:
        return (self.dy * self.grid_km, self.dx * self.grid_km)

    @property
    def aligned(self) -> bool:
        """Within one pixel is alignment; convective rain genuinely lags the
        cloud top by a little, so demanding an exact zero would be wrong."""
        return abs(self.dy) <= 1 and abs(self.dx) <= 1

    def report(self) -> str:
        ky, kx = self.offset_km
        lines = [
            f"peak agreement at offset (dy={self.dy:+d}, dx={self.dx:+d}) px "
            f"= ({ky:+.0f}, {kx:+.0f}) km",
            f"  score at peak {self.peak_score:.4f}, at zero offset {self.zero_score:.4f}",
        ]
        if self.aligned:
            lines.append("  ALIGNED (within one 4 km pixel)")
        else:
            lines.append(
                f"  MISALIGNED -- a systematic {max(abs(ky), abs(kx)):.0f} km shift. "
                f"Do NOT build training data until this is resolved: the model "
                f"would learn a stable, wrong cloud-to-rain displacement and "
                f"every score would still look healthy.")
        return "\n".join(lines)


def alignment_offset(
    tir_k: np.ndarray,
    rain_mmhr: np.ndarray,
    tir_cold_k: float = 240.0,
    rain_mm: float = 0.5,
    max_offset: int = 10,
    grid_km: float = 4.0,
) -> AlignmentResult:
    """Find the pixel offset at which cold cloud best overlaps rain.

    tir_cold_k default 240 K picks out deep convective tops; rain_mm 0.5
    picks out meaningful rain rather than drizzle. Both are thresholds on
    fields that must already be on the SAME grid.
    """
    tir = np.asarray(tir_k, dtype=np.float64)
    rain = np.asarray(rain_mmhr, dtype=np.float64)
    if tir.shape != rain.shape:
        raise ValueError(f"fields must be on the same grid: {tir.shape} vs {rain.shape}")

    cold = np.isfinite(tir) & (tir <= tir_cold_k)
    wet = np.isfinite(rain) & (rain >= rain_mm)
    if not cold.any():
        raise ValueError(f"no pixels colder than {tir_cold_k} K -- no convection "
                         f"in this scan, or the TIR field is not in kelvin")
    if not wet.any():
        raise ValueError(f"no pixels wetter than {rain_mm} mm/hr -- nothing to "
                         f"align against in this scan")

    n = 2 * max_offset + 1
    scores = np.zeros((n, n))
    for i, dy in enumerate(range(-max_offset, max_offset + 1)):
        for j, dx in enumerate(range(-max_offset, max_offset + 1)):
            shifted = np.roll(np.roll(wet, dy, axis=0), dx, axis=1)
            inter = np.count_nonzero(cold & shifted)
            union = np.count_nonzero(cold | shifted)
            scores[i, j] = inter / union if union else 0.0   # IoU

    k = int(np.argmax(scores))
    iy, ix = divmod(k, n)
    return AlignmentResult(
        dy=iy - max_offset, dx=ix - max_offset,
        peak_score=float(scores[iy, ix]),
        zero_score=float(scores[max_offset, max_offset]),
        grid_km=grid_km, scores=scores, max_offset=max_offset,
    )
