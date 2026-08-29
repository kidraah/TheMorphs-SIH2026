"""THE GATE. Does accumulation from MERIT `dir` reproduce MERIT's own `upa`?

The D8 convention in nowcast_flood/flow.py is taken from documentation. A
transposed or reversed reading routes water confidently in the wrong
direction, still delineates basins, and still returns plausible discharges
and arrival times -- there is no symptom. Two independently derived layers
agreeing is the cheap proof, and `upa` is already on disk.

Nothing from the flood track should be quoted until this passes.

    python scripts/verify_flow_direction.py --tile n20e075
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np

MERIT = "/Users/evad/MERIT"
# Cell area is computed per ROW, not as a constant: longitude spacing shrinks
# as cos(lat), and across a 5-degree tile a constant-area assumption is a
# several-percent systematic bias in exactly the comparison being made.
#
# Exact spherical-cap area rather than dx*dy, on the GRS80 authalic radius:
#     A = dlambda * R^2 * (sin(phi_top) - sin(phi_bottom))
# The rectangular approximation is ~0.14% small at these latitudes, and area
# scales discharge linearly (q_p = 0.208 A Q / T_p), so it is not a rounding
# detail -- it lands directly in the flood numbers.
R_AUTHALIC_M = 6371007.181
CELL_DEG = 3.0 / 3600.0


def find(layer: str, tile: str) -> str:
    hits = glob.glob(os.path.join(MERIT, f"{layer}_*", f"{tile}_{layer}.tif"))
    if not hits:
        raise FileNotFoundError(f"no {layer} tile for {tile} under {MERIT}")
    return hits[0]


def row_area_km2(top_lat: float, n_rows: int) -> np.ndarray:
    """Cell area for each raster row, km^2. Exact on the authalic sphere."""
    phi_top = np.deg2rad(top_lat - np.arange(n_rows) * CELL_DEG)
    phi_bot = np.deg2rad(top_lat - (np.arange(n_rows) + 1) * CELL_DEG)
    dlam = np.deg2rad(CELL_DEG)
    return dlam * R_AUTHALIC_M ** 2 * (np.sin(phi_top) - np.sin(phi_bot)) / 1e6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tile", default="n20e075", help="e.g. n20e075")
    ap.add_argument("--window", type=int, default=0,
                    help="crop to NxN cells from the tile centre (0 = full)")
    ap.add_argument("--tol", type=float, default=0.02)
    ap.add_argument("--out", default="runs/flow_direction_gate.json")
    a = ap.parse_args()

    import rasterio

    from nowcast_flood.flow import build_flow_grid, verify_against_upa

    with rasterio.open(find("dir", a.tile)) as src:
        top_lat = src.transform.f
        if a.window:
            h, w = src.height, src.width
            off = ((h - a.window) // 2, (w - a.window) // 2)
            win = rasterio.windows.Window(off[1], off[0], a.window, a.window)
            direction = src.read(1, window=win)
            top_lat = src.transform.f + off[0] * src.transform.e
            row0 = off[0]
        else:
            direction = src.read(1)
            row0 = 0
        print(f"dir  {direction.shape} {direction.dtype} "
              f"top_lat {top_lat:.4f}")

    with rasterio.open(find("upa", a.tile)) as src:
        upa = (src.read(1, window=win) if a.window else src.read(1)).astype(np.float64)
        upa_nodata = src.nodata
    upa = np.where((upa == upa_nodata) | (upa <= 0), np.nan, upa)

    codes, counts = np.unique(direction, return_counts=True)
    print("direction codes present:")
    for c, n in zip(codes, counts):
        print(f"    {int(c):>7}  {n:>12,}  {100*n/direction.size:>6.2f}%")

    # Ocean/nodata cells cannot route. Treat them as terminals rather than
    # letting a negative code fall through as "no match" -- silently, they
    # would become sinks anyway, but not visibly.
    d = np.where(direction > 0, direction, 0).astype(np.int32)

    fg = build_flow_grid(d)
    area = np.broadcast_to(row_area_km2(top_lat, d.shape[0])[:, None],
                           d.shape).copy()
    res = verify_against_upa(fg, upa, area, tol=a.tol)

    print(f"\n{'GATE':>26}: {'PASS' if res['passed'] else 'FAIL'}")
    for k in ("fraction_within_tol", "median_rel_error", "p99_rel_error",
              "scale_offset", "scale_spread",
              "median_rel_error_after_rescale", "n_compared",
              "boundary_excluded"):
        print(f"{k:>26}: {res[k]}")
    print(f"{'verdict':>26}: {res['verdict']}")
    if res["reason"]:
        print(f"\n  {res['reason']}")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"tile": a.tile, "window": a.window,
                                       **res}, indent=2, default=float))
    print(f"\nwrote {a.out}")
    return 0 if res["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
