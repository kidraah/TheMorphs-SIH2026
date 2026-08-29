"""GPM IMERG ingestion onto the common 4 km grid.

IMERG is the LABEL source (docs/LABELS.md), so correctness matters more here
than anywhere else in the pipeline: an input bug degrades the model, a label
bug teaches it the wrong thing while every score looks healthy.

Three specifics that bite
-------------------------
1. **Axis order.** IMERG HDF5 stores grids as (time, lon, lat) -- longitude
   FIRST. Almost every other gridded product is (time, lat, lon). Reading it
   without transposing produces a transposed rain field that still looks like
   weather, and it will not resample into an obvious error: it will just put
   rain in the wrong place. `read_precipitation` transposes and
   `check_physics` re-checks the shape against the known 0.1 deg geometry.

2. **Fill value.** Missing is -9999.9, not NaN. Left alone it becomes a
   large negative rain rate that survives thresholding as "no rain" and
   poisons any mean.

3. **Variable name changed between versions.** V06 used
   `Grid/precipitationCal`; V07 renamed it `Grid/precipitation`. Both are
   tried, and the one found is reported.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .grids import india_area
from .sentinels import detect_pileups

FILL_VALUE = -9999.9
IMERG_RESOLUTION_DEG = 0.1
IMERG_SHAPE_LONLAT = (3600, 1800)          # (lon, lat) as stored
PRECIP_VARIANTS = ("Grid/precipitation", "Grid/precipitationCal")

# mm/hr. The world record for hourly rainfall is ~305 mm; anything above a
# few hundred is a decoding problem, not weather.
EXPECTED_RATE_MMHR = (0.0, 400.0)


@dataclass
class ImergScan:
    rate_mmhr: np.ndarray       # (lat, lon) after transpose
    lats: np.ndarray
    lons: np.ndarray
    variable: str
    path: str
    checks: list = field(default_factory=list)


def _grid_coords():
    """Cell-centre coordinates of the global 0.1 deg IMERG grid."""
    lons = -180.0 + IMERG_RESOLUTION_DEG * (np.arange(IMERG_SHAPE_LONLAT[0]) + 0.5)
    lats = -90.0 + IMERG_RESOLUTION_DEG * (np.arange(IMERG_SHAPE_LONLAT[1]) + 0.5)
    return lats, lons


def read_precipitation(path) -> ImergScan:
    """Read one IMERG half-hourly file into (lat, lon) mm/hr with NaN fill."""
    import h5py

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"no IMERG file at {p}")

    with h5py.File(p, "r") as fh:
        var = next((v for v in PRECIP_VARIANTS if v in fh), None)
        if var is None:
            grid = list(fh["Grid"].keys()) if "Grid" in fh else list(fh.keys())
            raise KeyError(
                f"none of {PRECIP_VARIANTS} in {p.name}. Present under Grid: "
                f"{grid[:15]}. IMERG renamed precipitationCal -> precipitation "
                f"at V07; add the new name to PRECIP_VARIANTS.")
        arr = np.asarray(fh[var][:], dtype=np.float64)

    arr = arr[0] if arr.ndim == 3 else arr        # drop the length-1 time axis
    if arr.shape != IMERG_SHAPE_LONLAT:
        raise ValueError(
            f"expected (lon, lat) = {IMERG_SHAPE_LONLAT} for a 0.1 deg global "
            f"grid, got {arr.shape}. If this is a subset or a different "
            f"resolution the transpose below would silently mis-orient it.")

    arr = arr.T                                    # -> (lat, lon)
    arr[arr <= FILL_VALUE + 1e-3] = np.nan
    lats, lons = _grid_coords()
    return ImergScan(rate_mmhr=arr, lats=lats, lons=lons, variable=var, path=str(p))


def check_physics(scan: ImergScan, min_valid_frac: float = 0.5) -> list:
    """Rain rates must be non-negative and physically bounded.

    Coverage is a FRACTION threshold, not merely "some finite pixel exists".
    A real IMERG scan is near-fully populated -- dry is 0.0, not missing -- so
    a mostly-empty grid means a bad read or a subset masquerading as global.
    """
    out = []
    a = scan.rate_mmhr
    finite = np.isfinite(a)
    frac = float(finite.mean())
    out.append(("coverage", frac >= min_valid_frac,
                f"{100 * frac:.1f}% finite (need >= {100 * min_valid_frac:.0f}%)"))
    if finite.any():
        lo, hi = float(np.nanmin(a)), float(np.nanmax(a))
        elo, ehi = EXPECTED_RATE_MMHR
        out.append(("rate range", lo >= elo and hi <= ehi,
                    f"{lo:.2f}-{hi:.2f} mm/hr (expected {elo:.0f}-{ehi:.0f})"))
        out.append(("no negative rain", lo >= 0.0, f"min {lo:.3f} mm/hr"))
    out.append(("orientation", a.shape == IMERG_SHAPE_LONLAT[::-1],
                f"shape {a.shape} (lat, lon)"))

    # Standing sentinel audit -- see nowcast_data/sentinels.py. IMERG's -9999.9
    # is already handled, but assume there is another one we have not met.
    pu = detect_pileups(a, min_fraction=0.001)
    bad = [p for p in pu.pileups if p.suspicion == "high"]
    out.append(("value pile-up", not bad,
                ("; ".join(f"{p.value:g} at {100 * p.fraction:.2f}%" for p in bad)
                 + "  <-- check the product docs") if bad else "no isolated spikes"))
    return out


def resample_to_grid(scan: ImergScan, radius_of_influence: float = 15000.0,
                     resampler: str = "nearest") -> np.ndarray:
    """Put IMERG on the common 4 km grid.

    Nearest, not bilinear: an IMERG cell is ~11 km and the target is 4 km, so
    the honest result is blocky. Interpolating would manufacture smooth rain
    gradients at scales IMERG cannot resolve -- exactly the error avoided on
    the input side with water vapour.
    """
    from pyresample.geometry import GridDefinition
    from pyresample.kd_tree import resample_nearest

    lon2d, lat2d = np.meshgrid(scan.lons, scan.lats)
    src = GridDefinition(lons=lon2d, lats=lat2d)
    filled = np.where(np.isfinite(scan.rate_mmhr), scan.rate_mmhr, -1.0)

    out = resample_nearest(src, filled, india_area(),
                           radius_of_influence=radius_of_influence,
                           fill_value=-1.0)
    return np.where(out < 0, np.nan, out).astype(np.float32)


def ingest(path, strict: bool = True) -> tuple[np.ndarray, list]:
    scan = read_precipitation(path)
    checks = check_physics(scan)
    failed = [c for c in checks if not c[1]]
    if strict and failed:
        raise ValueError("IMERG failed sanity checks:\n" +
                         "\n".join(f"  {n}: {d}" for n, _, d in failed))
    return resample_to_grid(scan), checks


def fetch(*_args, **_kw):
    """Not implemented -- see nowcast_data.insat.fetch_scan for the reasoning.

    GES DISC needs an Earthdata Login plus a one-time application
    authorisation, and the archive path embeds the product version, which
    changes. Download a day by hand from
    https://disc.gsfc.nasa.gov (credentials in .env), then pass local paths
    to `ingest`.
    """
    raise NotImplementedError(
        "IMERG fetch is not implemented -- download from GES DISC and pass "
        "local paths to ingest(). Requires EARTHDATA_USERNAME/PASSWORD and a "
        "one-time 'NASA GESDISC DATA ARCHIVE' app authorisation.")
