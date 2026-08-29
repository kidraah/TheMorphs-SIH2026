"""Who is actually exposed: low HAND, high upstream runoff, and people there.

Three conditions, and all three are needed. Low ground with no water is not
a flood; a lot of water in a deep gorge is not a disaster; and both together
in an empty valley is not an alert anybody should be woken for. Scoring only
the first two is how a flood product ends up ranking uninhabited Himalayan
headwaters above Mumbai.

HAND (height above nearest drainage) is the right terrain variable here
rather than elevation or slope: it is the vertical distance a cell sits
above the stream it drains to, so it says directly how much stage rise it
takes to inundate that cell. It is already on disk (MERIT `hnd`).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ExposureResult:
    exposed: np.ndarray          # (H, W) bool
    people: np.ndarray | None    # (B,) exposed population per basin, if given
    area_km2: np.ndarray         # (B,) exposed area per basin
    hand_threshold_m: float
    runoff_threshold_mm: float

    def report(self, basins=None, top: int = 10) -> str:
        lines = [f"exposure: HAND <= {self.hand_threshold_m:g} m and basin "
                 f"runoff >= {self.runoff_threshold_mm:g} mm"]
        if self.people is None:
            lines.append("    no settlement layer supplied -- AREA ONLY. Ranking "
                         "basins by exposed area puts empty headwaters above "
                         "cities; see docs/FLOOD_DATA.md for GHS-POP.")
        tot = float(np.nansum(self.area_km2))
        lines.append(f"    {tot:,.0f} km^2 exposed across "
                     f"{int(np.sum(self.area_km2 > 0))} basins")
        if self.people is not None:
            lines.append(f"    {np.nansum(self.people):,.0f} people exposed")
        return "\n".join(lines)


def flood_exposure(hand_m, basin_runoff_mm, basins,
                   hand_threshold_m: float = 5.0,
                   runoff_threshold_mm: float = 25.0,
                   population=None, cell_area_km2=1.0) -> ExposureResult:
    """Cells that are low-lying AND in a basin producing a lot of runoff.

    `basin_runoff_mm` is per-basin (length B); it is broadcast back onto the
    grid through the basin labels, which is the point -- the runoff that
    threatens a cell is the runoff of the basin upstream of it, not the rain
    that fell on the cell itself. That distinction is the entire difference
    between a flash-flood product and a rainfall map.
    """
    hand = np.asarray(hand_m, dtype=np.float64)
    lab = basins.labels
    idx = np.searchsorted(basins.ids, lab.clip(0))
    valid_lab = lab >= 0

    runoff = np.asarray(basin_runoff_mm, dtype=np.float64)
    if runoff.shape != basins.ids.shape:
        raise ValueError(f"basin_runoff_mm has {runoff.shape}, expected "
                         f"{basins.ids.shape} (one value per basin)")
    on_grid = np.where(valid_lab, runoff[idx], np.nan)

    exposed = (np.isfinite(hand) & (hand <= hand_threshold_m)
               & np.isfinite(on_grid) & (on_grid >= runoff_threshold_mm))

    area_cell = (np.full(lab.shape, float(cell_area_km2)) if np.isscalar(cell_area_km2)
                 else np.asarray(cell_area_km2, dtype=np.float64))
    sel = exposed & valid_lab
    area = np.bincount(idx[sel], weights=area_cell[sel], minlength=basins.n)

    people = None
    if population is not None:
        pop = np.asarray(population, dtype=np.float64)
        ok = sel & np.isfinite(pop)
        people = np.bincount(idx[ok], weights=pop[ok], minlength=basins.n)
    return ExposureResult(exposed, people, area, float(hand_threshold_m),
                          float(runoff_threshold_mm))
