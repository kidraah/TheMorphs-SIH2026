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
        # Prefer the file's OWN coordinate arrays over computed ones. They
        # agree exactly here, but computing them bakes in an assumption about
        # grid origin and cell registration that a future product version
        # could change silently -- and a half-cell longitude error is a ~5 km
        # uniform shift, indistinguishable from a georeferencing bug.
        file_lats = np.asarray(fh["Grid/lat"][:], dtype=np.float64) if "Grid/lat" in fh else None
        file_lons = np.asarray(fh["Grid/lon"][:], dtype=np.float64) if "Grid/lon" in fh else None
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
    if file_lats is not None and file_lons is not None:
        if file_lats.size != lats.size or file_lons.size != lons.size:
            raise ValueError(
                f"file grid {file_lons.size}x{file_lats.size} does not match the "
                f"expected 0.1deg global grid {lons.size}x{lats.size}")
        drift = max(float(np.max(np.abs(file_lats - lats))),
                    float(np.max(np.abs(file_lons - lons))))
        if drift > 0.01:
            raise ValueError(
                f"file coordinates differ from the assumed 0.1deg grid by "
                f"{drift:.4f} deg (~{drift * 111:.1f} km). Using the computed grid "
                f"would apply a uniform shift indistinguishable from a "
                f"georeferencing error.")
        lats, lons = file_lats, file_lons
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


GES_DISC_ROOT = "https://gpm1.gesdisc.eosdis.nasa.gov/data/GPM_L3"
IMERG_FINAL = "GPM_3IMERGHH.07"


class _EarthdataSession:
    """requests.Session that survives the GES DISC -> URS redirect.

    GES DISC redirects to urs.earthdata.nasa.gov to authenticate, then back.
    A plain Session re-sends the Authorization header to the redirected host,
    which the archive rejects. This is NASA's documented workaround: drop the
    header on a cross-host redirect unless one end is the auth host.
    """

    AUTH_HOST = "urs.earthdata.nasa.gov"

    def __new__(cls, username, password):
        import requests

        class S(requests.Session):
            def rebuild_auth(self, prepared_request, response):
                headers = prepared_request.headers
                if "Authorization" not in headers:
                    return
                orig = requests.utils.urlparse(response.request.url).hostname
                new = requests.utils.urlparse(prepared_request.url).hostname
                if orig != new and new != cls.AUTH_HOST and orig != cls.AUTH_HOST:
                    del headers["Authorization"]

        s = S()
        s.auth = (username, password)
        return s


def granule_url(when, product: str = IMERG_FINAL, version: str = "V07B") -> str:
    """Archive URL for one half-hourly granule.

    Filenames encode the half-open window and a minutes-since-midnight index,
    e.g. ...-S234000-E235959.1410.V07B.HDF5 for 23:40-23:59. Getting that
    index wrong yields a 404, not a wrong file.
    """
    import pandas as pd

    t = pd.Timestamp(when)
    start = t.floor("30min")
    end = start + pd.Timedelta(minutes=29, seconds=59)
    minutes = start.hour * 60 + start.minute
    name = (f"3B-HHR.MS.MRG.3IMERG.{start:%Y%m%d}"
            f"-S{start:%H%M%S}-E{end:%H%M%S}.{minutes:04d}.{version}.HDF5")
    return f"{GES_DISC_ROOT}/{product}/{start:%Y}/{start.dayofyear:03d}/{name}"


def fetch(when, dest_dir, product: str = IMERG_FINAL, version: str = "V07B",
          overwrite: bool = False, timeout: int = 300):
    """Download one IMERG granule. Returns the local path.

    Requires EARTHDATA_USERNAME / EARTHDATA_PASSWORD (see .env) AND a one-time
    authorisation of "NASA GESDISC DATA ARCHIVE" in the Earthdata profile --
    without it the archive returns a redirect loop rather than a clear 401.
    """
    from pathlib import Path as _P

    from .credentials import get

    url = granule_url(when, product, version)
    dest = _P(dest_dir) / url.rsplit("/", 1)[-1]
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not overwrite and dest.stat().st_size > 0:
        return dest

    session = _EarthdataSession(get("EARTHDATA_USERNAME"), get("EARTHDATA_PASSWORD"))
    r = session.get(url, stream=True, timeout=timeout)
    if r.status_code == 401:
        raise PermissionError(
            f"401 from GES DISC. Credentials are set, so this is almost always "
            f"the missing app authorisation: Earthdata profile -> Applications "
            f"-> Authorized Apps -> approve 'NASA GESDISC DATA ARCHIVE'.")
    if r.status_code == 404:
        raise FileNotFoundError(
            f"404 for {url}\nFinal-run IMERG lags ~3.5 months; for recent dates "
            f"use GPM_3IMERGHHL (Late) or GPM_3IMERGHHE (Early) -- but do NOT "
            f"mix runs within one training set, their biases differ.")
    r.raise_for_status()

    tmp = dest.with_suffix(dest.suffix + ".part")
    with open(tmp, "wb") as fh:
        for chunk in r.iter_content(chunk_size=1 << 20):
            fh.write(chunk)
    tmp.replace(dest)
    return dest


def fetch_day(date, dest_dir, times=None, **kw) -> list:
    """Fetch granules for one day. `times` limits to specific HH:MM strings."""
    import pandas as pd

    d = pd.Timestamp(date).normalize()
    slots = ([d + pd.Timedelta(minutes=30 * i) for i in range(48)] if times is None
             else [pd.Timestamp(f"{d:%Y-%m-%d} {t}") for t in times])
    out = []
    for t in slots:
        try:
            out.append(fetch(t, dest_dir, **kw))
        except Exception as e:
            print(f"  {t:%H:%M} FAILED {type(e).__name__}: {str(e)[:110]}")
    return out
