"""Deterministic flash-flood routing: predicted rain -> basin flood risk.

No model, no GPU, no MOSDAC. The transformer predicts rainfall; this turns
rainfall into a flood, by physics that can be checked against a
hydrologist's spreadsheet.

    from nowcast_flood import FloodRouter
    router = FloodRouter.from_layers(direction=..., upa_km2=..., ...)
    fc = router.route(rain_mm, rain_duration_min=60)

See docs/FLOOD_DATA.md for the layers this needs and where they come from.
"""
from .basins import Basins, aggregate, delineate, properties
from .curve_number import (WORLDCOVER_CN, adjust_amc, curve_number,
                           potential_retention, runoff_depth)
from .exposure import ExposureResult, flood_exposure
from .flow import FlowGrid, build_flow_grid, flow_accumulate, verify_against_upa
from .route import FloodForecast, FloodRouter
from .timing import (peak_discharge, time_of_concentration, time_to_peak)

__all__ = ["FloodRouter", "FloodForecast", "Basins", "FlowGrid",
           "build_flow_grid", "flow_accumulate", "verify_against_upa",
           "delineate", "properties", "aggregate", "curve_number",
           "adjust_amc", "runoff_depth", "potential_retention",
           "WORLDCOVER_CN", "time_of_concentration", "time_to_peak",
           "peak_discharge", "flood_exposure", "ExposureResult"]
