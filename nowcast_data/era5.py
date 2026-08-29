"""ERA5 thermodynamics from ARCO-ERA5 (public Zarr on Google Cloud).

Why ARCO and not the CDS download route
---------------------------------------
ARCO-ERA5 is a public, analysis-ready Zarr store: no account, no approval
queue, no request-and-wait, and — the part that matters — **lazily
sliceable**. We read the India box for the hours we want instead of
downloading globally and cropping. The CDS route would mean queued requests
returning GRIB that then needs converting.

Why ERA5 and not IMDAA: see docs/DATA_DECISIONS.md. Short version, IMDAA is
12 km hourly against ERA5's ~25 km but ends ~2018-2020, while our workhorse
satellite 3DR runs 2016 to present — using it would put a thermodynamic
discontinuity in the middle of the training set.

What comes precomputed vs what we derive
----------------------------------------
Precomputed by ECMWF, and we take them rather than rederiving:

    CAPE, CIN                        the instability term
    total_column_water_vapour        the moisture term (IWV)
    vertical integrals of vapour flux

Deriving CAPE ourselves from multi-level profiles would mean adopting our own
parcel assumptions (surface vs mixed-layer, virtual temperature correction,
ice phase) and getting a number that disagrees with every published ERA5
figure for no benefit.

Derived here, because ERA5 does not ship them:

    bulk wind shear          |V(500) - V(850)| — storm organisation
    low-level convergence    -div(V) at 850 hPa — the lift trigger
    moisture convergence     -div(vertical vapour flux) — the fuel supply

Derivatives are taken on the sphere with the cos(latitude) metric. Doing them
in raw degrees would understate the zonal gradient by ~8% at Kanyakumari and
~18% at Kashmir — a latitude-dependent bias in exactly the term that
identifies where storms initiate.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

ARCO_URL = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"

# India analysis box, a little wider than the 4 km grid so derivatives at the
# edge have neighbours rather than one-sided stencils.
INDIA_BOX = {"lat": (3.0, 42.0), "lon": (60.0, 104.0)}

SINGLE_LEVEL = ("convective_available_potential_energy",
                "convective_inhibition",
                "total_column_water_vapour",
                "vertical_integral_of_eastward_water_vapour_flux",
                "vertical_integral_of_northward_water_vapour_flux")
MULTI_LEVEL = ("u_component_of_wind", "v_component_of_wind",
               "specific_humidity", "temperature")

SHEAR_LEVELS = (850, 500)      # hPa: low-level flow vs mid-level steering
CONVERGENCE_LEVEL = 850        # hPa: where low-level convergence is diagnosed

EARTH_RADIUS_M = 6_371_000.0

# Physically plausible ranges, for the same reason INSAT has them: a wrong
# unit or a mis-sliced level does not raise, it returns a plausible array.
EXPECTED = {
    "cape": (0.0, 8000.0),            # J/kg; >6000 is exceptional but real
    # ERA5 reports CIN as a POSITIVE magnitude, not a signed quantity.
    # Verified on 2023-07-09: 0% of finite values are negative, max 998.8.
    # An earlier version of this check asserted CIN <= 0 and failed valid data.
    "cin": (0.0, 2000.0),
    "tcwv": (0.0, 100.0),             # kg/m^2
    "shear": (0.0, 100.0),            # m/s
    "convergence": (-5e-3, 5e-3),     # 1/s
    "moisture_convergence": (-5e-2, 5e-2),   # kg/m^2/s
}


def open_store(url: str = ARCO_URL):
    """Open ARCO-ERA5 lazily. Nothing is downloaded until a slice is computed."""
    import xarray as xr
    return xr.open_zarr(url, chunks=None, storage_options={"token": "anon"})


def _subset(ds, time, box=None, levels=None):
    box = box or INDIA_BOX
    lat0, lat1 = box["lat"]
    lon0, lon1 = box["lon"]
    # ERA5 latitude runs north -> south, so the slice is reversed.
    sel = ds.sel(time=time, latitude=slice(lat1, lat0), longitude=slice(lon0, lon1))
    if levels is not None:
        sel = sel.sel(level=list(levels))
    return sel


def _spherical_divergence(u, v, lat_deg, lon_deg):
    """div(V) on a lat/lon grid, in 1/s.

    The cos(lat) metric is not optional: without it the zonal derivative is
    wrong by 1/cos(lat), which over India ranges from ~1.01 at Kanyakumari to
    ~1.22 at Kashmir. That would be a latitude-dependent bias in the
    convergence field — a systematic error that looks like real geography.
    """
    lat = np.deg2rad(np.asarray(lat_deg, dtype=np.float64))
    lon = np.deg2rad(np.asarray(lon_deg, dtype=np.float64))
    coslat = np.cos(lat)[:, None]

    dlon = np.gradient(lon)[None, :]
    dlat = np.gradient(lat)[:, None]
    dx = EARTH_RADIUS_M * coslat * dlon
    dy = EARTH_RADIUS_M * dlat

    du_dx = np.gradient(u, axis=1) / dx
    # the metric term: d(v cos(lat))/dy / cos(lat)
    dv_dy = np.gradient(v * coslat, axis=0) / dy / np.maximum(coslat, 1e-6)
    return du_dx + dv_dy


@dataclass
class ERA5Fields:
    time: str
    lat: np.ndarray
    lon: np.ndarray
    fields: dict = field(default_factory=dict)

    def __getitem__(self, k):
        return self.fields[k]

    @property
    def names(self):
        return sorted(self.fields)


def load_fields(time, box=None, url: str = ARCO_URL, ds=None) -> ERA5Fields:
    """Precomputed + derived thermodynamic fields for one timestamp."""
    ds = ds if ds is not None else open_store(url)
    box = box or INDIA_BOX

    sl = _subset(ds[list(SINGLE_LEVEL)], time, box)
    lat = np.asarray(sl.latitude.values, dtype=np.float64)
    lon = np.asarray(sl.longitude.values, dtype=np.float64)

    out = {
        "cape": np.asarray(sl["convective_available_potential_energy"].values),
        "cin": np.asarray(sl["convective_inhibition"].values),
        "tcwv": np.asarray(sl["total_column_water_vapour"].values),
    }

    # Moisture convergence from the vertically integrated flux. Preferred over
    # deriving it from q and wind: this is what the model's own mass-consistent
    # integration produced, so it does not inherit our discretisation error.
    qu = np.asarray(sl["vertical_integral_of_eastward_water_vapour_flux"].values)
    qv = np.asarray(sl["vertical_integral_of_northward_water_vapour_flux"].values)
    out["moisture_convergence"] = -_spherical_divergence(qu, qv, lat, lon)

    ml = _subset(ds[["u_component_of_wind", "v_component_of_wind"]], time, box,
                 levels=SHEAR_LEVELS)
    u = np.asarray(ml["u_component_of_wind"].values)
    v = np.asarray(ml["v_component_of_wind"].values)
    lo, hi = list(SHEAR_LEVELS).index(SHEAR_LEVELS[0]), list(SHEAR_LEVELS).index(SHEAR_LEVELS[1])
    out["shear"] = np.hypot(u[hi] - u[lo], v[hi] - v[lo])

    conv = _subset(ds[["u_component_of_wind", "v_component_of_wind"]], time, box,
                   levels=(CONVERGENCE_LEVEL,))
    cu = np.asarray(conv["u_component_of_wind"].values).squeeze()
    cv = np.asarray(conv["v_component_of_wind"].values).squeeze()
    out["convergence"] = -_spherical_divergence(cu, cv, lat, lon)

    return ERA5Fields(time=str(time), lat=lat, lon=lon, fields=out)


def check_physics(f: ERA5Fields) -> list:
    """Range checks plus the standing sentinel scan."""
    from .sentinels import detect_pileups

    out = []
    for name, arr in f.fields.items():
        a = np.asarray(arr, dtype=np.float64)
        fin = np.isfinite(a)
        # CIN is legitimately undefined where there is no convective parcel to
        # inhibit -- ~40% of an India scene. That is a physical statement, not
        # missing data, so it is not held to the coverage bar. See
        # CIN_UNDEFINED_MEANS_NO_PARCEL below.
        min_cov = 0.0 if name == "cin" else 0.99
        out.append((f"{name} coverage", bool(fin.mean() >= min_cov),
                    f"{100 * fin.mean():.1f}% finite"
                    + (" (undefined where no convective parcel -- expected)"
                       if name == "cin" else "")))
        if not fin.any():
            continue
        v = a[fin]
        lo, hi = EXPECTED[name]
        ok = float(v.min()) >= lo and float(v.max()) <= hi
        out.append((f"{name} range", ok,
                    f"{v.min():.4g} .. {v.max():.4g} (expected {lo:g} .. {hi:g})"))
        pu = detect_pileups(v, min_fraction=0.001)
        bad = [p for p in pu.pileups if p.suspicion == "high"]
        out.append((f"{name} value pile-up", not bad,
                    "; ".join(f"{p.value:g} at {100*p.fraction:.2f}%" for p in bad)
                    or "no isolated spikes"))
    # ERA5 CIN is a positive magnitude.
    if "cin" in f.fields:
        c = f.fields["cin"][np.isfinite(f.fields["cin"])]
        if c.size:
            out.append(("cin sign convention", bool(c.min() >= 0.0),
                        f"min {c.min():.2f} J/kg (ERA5 CIN is a positive magnitude)"))
    # The NaN pattern must track low CAPE. If CIN were NaN at HIGH CAPE that
    # would be genuine missing data and the fill policy below would be unsafe.
    if "cin" in f.fields and "cape" in f.fields:
        cin, cape = f.fields["cin"], f.fields["cape"]
        nan = ~np.isfinite(cin)
        if nan.any() and np.isfinite(cin).any():
            m_nan, m_fin = float(cape[nan].mean()), float(cape[~nan].mean())
            out.append(("cin undefined tracks low CAPE", m_nan < m_fin,
                        f"mean CAPE {m_nan:.0f} where CIN undefined vs "
                        f"{m_fin:.0f} where defined"))
    return out


# The most dangerous fill in this dataset.
#
# CIN is NaN where ERA5 found no convective parcel. Filling that with 0.0 --
# the obvious default -- tells the model "zero inhibition", which reads as
# FAVOURABLE for convection. The truth is the opposite: no parcel means
# convection is not possible at all.
#
# Verified on 2023-07-09 over India: mean CAPE is 67 J/kg where CIN is
# undefined and 930 J/kg where it is defined, so the undefined cells are
# precisely the non-convective ones.
#
# Fill with a large inhibition instead, and carry a validity channel so the
# network can tell "strongly capped" from "no parcel at all".
CIN_UNDEFINED_MEANS_NO_PARCEL = True
CIN_NO_PARCEL_FILL = 2000.0      # J/kg -- effectively "convection impossible"


def fill_cin(cin: np.ndarray) -> tuple:
    """(filled, valid_mask). NEVER fill CIN with zero -- see above."""
    a = np.asarray(cin, dtype=np.float64)
    valid = np.isfinite(a)
    return np.where(valid, a, CIN_NO_PARCEL_FILL), valid


def resample_to_grid(f: ERA5Fields, names=None, radius_of_influence=40000.0) -> dict:
    """Put the fields on the common India 4 km grid.

    Nearest, not bilinear: ERA5 is ~25 km and the target 4 km, so the honest
    result is blocky. Interpolating would manufacture gradients at scales the
    reanalysis cannot resolve — the same argument applied to IMERG labels and
    to INSAT water vapour.
    """
    from pyresample.geometry import GridDefinition
    from pyresample.kd_tree import resample_nearest

    from .grids import india_area

    lon2d, lat2d = np.meshgrid(f.lon, f.lat)
    src = GridDefinition(lons=lon2d, lats=lat2d)
    area = india_area()
    out = {}
    for name in (names or f.names):
        a = np.asarray(f[name], dtype=np.float64)
        filled = np.where(np.isfinite(a), a, -9e9)
        r = resample_nearest(src, filled, area,
                             radius_of_influence=radius_of_influence,
                             fill_value=-9e9)
        out[name] = np.where(r < -8e9, np.nan, r).astype(np.float32)
    return out
