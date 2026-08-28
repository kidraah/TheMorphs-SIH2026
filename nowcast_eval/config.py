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
