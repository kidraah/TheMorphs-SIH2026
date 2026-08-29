"""Every judgment call in the harness, in one place.

Nothing here is math. These are *choices* -- and they move the reported
numbers more than any formula does. They live in a dataclass so that a
result can never be reported without the settings that produced it.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Sequence


@dataclass(frozen=True)
class EvalConfig:
    # --- Binarising the forecast -------------------------------------------
    # CSI/POD/FAR require a yes/no forecast, but the model emits probabilities.
    # Reporting a single threshold invites cherry-picking, so we sweep and
    # report the whole curve. `headline_threshold` is only for the summary row.
    thresholds: Sequence[float] = (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
    headline_threshold: float = 0.5

    # --- Neighbourhood sizes for FSS ---------------------------------------
    # In km. Converted to pixels with `grid_km`. A 4 km grid and a 50 km
    # neighbourhood gives a ~13x13 window.
    neighborhood_km: Sequence[float] = (0.0, 25.0, 50.0, 100.0)
    grid_km: float = 4.0

    # --- Geometry ----------------------------------------------------------
    # "grid"  -> (N, L, H, W); neighbourhood metrics (FSS) are meaningful.
    # "point" -> (N, L, S) at S irregular station locations; FSS is SKIPPED.
    # "basin" -> (N, L, B) over B sub-basin polygons; FSS is SKIPPED and
    #            elements are AREA-WEIGHTED (see `basin_weighting`).
    #
    # Flash floods are basin-scale, so the flood head genuinely scores on a
    # different geometry from the other two. It would "work" today declared
    # as point -- the array rank matches -- and that is the trap: point
    # geometry weights every element equally, which is right for gauges and
    # wrong for basins, because sub-basin areas differ by orders of
    # magnitude. Scoring them equally measures skill per basin instead of
    # per unit of land, flattering a forecast that gets many tiny headwater
    # basins right and under-counting the large valleys where people live.
    #
    # This is not a formality. Station data shoehorned into a degenerate
    # (N, L, 1, S) grid runs happily and reports an FSS that averaged over
    # adjacent station INDICES -- alphabetical-order smoothing wearing a
    # kilometre label. Declaring the geometry makes that impossible.
    geometry: str = "grid"

    # How basin elements are weighted in the contingency table. "area" is
    # the default and the defensible one; "equal" is available because a
    # per-basin skill number is sometimes what is wanted, but it must be a
    # stated choice rather than an accident of array shape. "population"
    # answers the question an alerting system actually cares about, and
    # needs a settlement layer.
    basin_weighting: str = "area"

    # --- Ground truth ------------------------------------------------------
    # If `obs` arrives continuous (e.g. mm/hr of QPE) this threshold makes it
    # binary. IMD's operational cloudburst figure is ~100 mm/hr over 20-30 km^2;
    # confirm per-hazard with a meteorologist before the main training run.
    obs_threshold: float | None = None

    # --- Missing data ------------------------------------------------------
    # True  -> masked cells are dropped from every statistic (recommended).
    # False -> masked cells are counted as no-event, which inflates correct
    #          negatives and quietly flatters FAR.
    drop_masked: bool = True

    # --- Probabilistic -----------------------------------------------------
    n_reliability_bins: int = 10

    # --- Provenance --------------------------------------------------------
    name: str = "default"
    notes: str = ""

    def __post_init__(self):
        if self.geometry not in ("grid", "point", "basin"):
            raise ValueError(f"geometry must be 'grid', 'point' or 'basin', "
                             f"got {self.geometry!r}")
        if self.basin_weighting not in ("area", "equal", "population"):
            raise ValueError(f"basin_weighting must be 'area', 'equal' or "
                             f"'population', got {self.basin_weighting!r}")

    @property
    def is_point(self) -> bool:
        return self.geometry == "point"

    @property
    def is_basin(self) -> bool:
        return self.geometry == "basin"

    @property
    def has_neighborhood(self) -> bool:
        """FSS is only meaningful on a regular grid.

        Point and basin geometries are both irregular, but for different
        reasons -- station array order carries no distance, and basin
        adjacency is a river network rather than a raster. Both skip FSS;
        neither should be silently reshaped into a grid to get one.
        """
        return self.geometry == "grid"

    def neighborhood_pixels(self) -> list[int]:
        """km -> odd pixel window widths."""
        out = []
        for km in self.neighborhood_km:
            n = max(1, int(round(km / self.grid_km)))
            if n % 2 == 0:
                n += 1
            out.append(n)
        return out

    def to_dict(self) -> dict:
        d = asdict(self)
        d["neighborhood_pixels"] = self.neighborhood_pixels()
        return d
