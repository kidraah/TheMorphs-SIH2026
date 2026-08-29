"""`route(rain_field) -> basin_risk, discharge, time_to_arrival`.

Deterministic physics, no model and no GPU: the transformer predicts rain,
this turns rain into a flood. Keeping the two apart is deliberate -- a
learned flood head trained on ~10 gauged events would be fitting noise,
while the routing is defensible from first principles and can be checked
against a hydrologist's spreadsheet.

The interface mirrors the model's on purpose: build once with the static
terrain, then call per rain field.

    router = FloodRouter.from_layers(direction=..., upa=..., hand=...,
                                     curve_number=..., elevation=...)
    fc = router.route(rain_mm, rain_duration_min=60)
    fc.basin_risk          # (B,) in [0, 1]
    fc.discharge_m3s       # (B,)
    fc.time_to_arrival_min # (B,)

Chain: rain -> SCS runoff depth per cell -> area-weighted mean per basin ->
SCS triangular peak discharge -> Kirpich arrival time -> risk.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import basins as _basins
from .curve_number import adjust_amc, runoff_depth
from .exposure import flood_exposure
from .flow import build_flow_grid, flow_accumulate
from .timing import (DEFAULT_METHOD, compare_methods, peak_discharge,
                     time_of_concentration)


@dataclass
class FloodForecast:
    basin_ids: np.ndarray
    basin_risk: np.ndarray            # (B,) in [0, 1]
    discharge_m3s: np.ndarray         # (B,)
    time_to_arrival_min: np.ndarray   # (B,)
    runoff_mm: np.ndarray             # (B,)
    # The spread across every t_c method is the honest uncertainty on a
    # warning lead time; a single number to the minute is not.
    arrival_low_min: np.ndarray = None
    arrival_high_min: np.ndarray = None
    exposure: object = None
    timing: object = None
    meta: dict = field(default_factory=dict)

    @property
    def n_basins(self) -> int:
        return len(self.basin_ids)

    def to_grid(self, labels: np.ndarray, values: np.ndarray | None = None):
        """Paint a per-basin vector back onto the analysis grid, for the map."""
        v = self.basin_risk if values is None else np.asarray(values)
        idx = np.searchsorted(self.basin_ids, labels.clip(0))
        return np.where(labels >= 0, v[idx], np.nan)

    def report(self, top: int = 5) -> str:
        ok = np.isfinite(self.discharge_m3s)
        lines = [f"flood forecast over {self.n_basins} basins "
                 f"({int(ok.sum())} with defined timing)"]
        if ok.any():
            order = np.argsort(-np.where(ok, self.basin_risk, -np.inf))[:top]
            lines.append(f"{'basin':>8} {'risk':>6} {'Q m3/s':>9} "
                         f"{'runoff mm':>10} {'arrival min':>12}")
            for i in order:
                lines.append(f"{int(self.basin_ids[i]):>8} "
                             f"{self.basin_risk[i]:>6.2f} "
                             f"{self.discharge_m3s[i]:>9.1f} "
                             f"{self.runoff_mm[i]:>10.1f} "
                             f"{self.time_to_arrival_min[i]:>12.0f}")
        if self.timing is not None:
            lines.append(self.timing.report())
        if self.exposure is not None:
            lines.append(self.exposure.report())
        return "\n".join(lines)


class FloodRouter:
    """Static terrain, prepared once. `route()` is the per-rain-field call."""

    def __init__(self, flow_grid, basins, curve_number, hand_m=None,
                 population=None, cell_area_km2=1.0, cell_size_m=1000.0,
                 amc: str = "II", channel: str = "natural",
                 channel_km2: float = 25.0, tc_method: str = DEFAULT_METHOD):
        self.fg = flow_grid
        self.basins = basins
        self.hand_m = hand_m
        self.population = population
        self.cell_area_km2 = cell_area_km2
        self.cell_size_m = cell_size_m
        self.channel = channel
        self.channel_km2 = channel_km2
        self.tc_method = tc_method
        # AMC is applied ONCE, here, and recorded. Applying it per call would
        # let a caller silently change the wettest-case assumption between
        # forecasts and compare numbers that are not comparable.
        self.amc = amc
        self.cn = adjust_amc(np.asarray(curve_number, dtype=np.float64), amc)

    # -- construction -------------------------------------------------------
    @classmethod
    def from_layers(cls, direction, upa_km2, curve_number, hand_m=None,
                    elevation_m=None, population=None, cell_area_km2=1.0,
                    cell_size_m=1000.0, channel_km2: float = 25.0,
                    amc: str = "II", channel: str = "natural",
                    tc_method: str = DEFAULT_METHOD):
        fg = build_flow_grid(np.asarray(direction))
        labels = _basins.delineate(fg, upa_km2, channel_km2=channel_km2)
        bas = _basins.properties(fg, labels, cell_area_km2, cell_size_m,
                                 upa_km2, elevation_m, channel_km2=channel_km2)
        return cls(fg, bas, curve_number, hand_m, population, cell_area_km2,
                   cell_size_m, amc, channel, channel_km2, tc_method)

    # -- the per-forecast call ---------------------------------------------
    def route(self, rain_mm, rain_duration_min: float = 60.0,
              risk_reference_m3s: float | None = None,
              hand_threshold_m: float = 5.0,
              runoff_threshold_mm: float = 25.0) -> FloodForecast:
        """Rain depth field (H, W) in mm -> per-basin flood forecast."""
        rain = np.asarray(rain_mm, dtype=np.float64)
        if rain.shape != self.basins.labels.shape:
            raise ValueError(f"rain {rain.shape} does not match the terrain "
                             f"grid {self.basins.labels.shape}")

        q_cell = runoff_depth(rain, self.cn)
        q_basin = _basins.aggregate(q_cell, self.basins, "mean",
                                    self.cell_area_km2)

        timing = time_of_concentration(
            self.basins.channel_length_m, self.basins.channel_slope,
            self.basins.area_km2, channel=self.channel,
            rain_duration_min=rain_duration_min, method=self.tc_method,
            mean_elev_above_outlet_m=self.basins.mean_elev_above_outlet_m)
        spread = compare_methods(
            self.basins.channel_length_m, self.basins.channel_slope,
            self.basins.area_km2, self.basins.mean_elev_above_outlet_m,
            rain_duration_min, self.channel)
        discharge = peak_discharge(q_basin, self.basins.area_km2,
                                   timing.t_peak_min)

        # Risk is a normalised discharge, and the reference is explicit.
        # A default drawn from the current field's own maximum would make
        # every forecast contain a 1.0 -- including a dry day.
        ref = (risk_reference_m3s if risk_reference_m3s is not None
               else self._default_reference())
        with np.errstate(invalid="ignore", divide="ignore"):
            risk = np.clip(discharge / ref, 0.0, 1.0)
        risk = np.where(np.isfinite(risk), risk, np.nan)

        exposure = None
        if self.hand_m is not None:
            exposure = flood_exposure(self.hand_m, q_basin, self.basins,
                                      hand_threshold_m, runoff_threshold_mm,
                                      self.population, self.cell_area_km2)
        return FloodForecast(
            basin_ids=self.basins.ids, basin_risk=risk,
            discharge_m3s=discharge, time_to_arrival_min=timing.t_peak_min,
            runoff_mm=q_basin, exposure=exposure, timing=timing,
            arrival_low_min=spread["t_peak_min_low"],
            arrival_high_min=spread["t_peak_min_high"],
            meta={"amc": self.amc, "channel": self.channel,
                  "tc_method": self.tc_method,
                  "channel_km2": self.channel_km2,
                  "rain_duration_min": rain_duration_min,
                  "risk_reference_m3s": float(ref),
                  "cell_area_km2": (None if not np.isscalar(self.cell_area_km2)
                                    else float(self.cell_area_km2))})

    def _default_reference(self) -> float:
        """A bankfull-scale discharge from basin area alone.

        q ~ 0.5 * A^0.8 m^3/s is a crude regional envelope, and it is a
        PLACEHOLDER in the same sense as the 0.80 FAR ceiling: it makes the
        risk scale defined rather than defensible. Replace it with CWC
        gauged bankfull discharges before any risk number is published.
        """
        a = np.asarray(self.basins.area_km2, dtype=np.float64)
        return float(np.nanmedian(0.5 * np.power(np.maximum(a, 1e-6), 0.8))) or 1.0
