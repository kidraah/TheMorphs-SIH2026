"""INSAT-3D/3DR L1B ingestion onto the common 4 km grid.

Read with satpy's `insat3d_img_l1b_h5` reader, resample with pyresample.
satpy 0.60 exposes VIS/SWIR/WV/MIR/TIR1/TIR2 and confirms the resolutions
this project was built around: VIS 1000 m, TIR1/TIR2 4000 m, WV 8000 m.
That asymmetry is why channels are matched individually rather than
uniformly resized.

Sanity checks come FIRST
------------------------
A reader that silently drifts from MOSDAC's current product format does not
raise -- it returns plausible-looking arrays in the wrong units, the wrong
calibration, or flipped geometry. `check_physics` is the cheap guard: real
TIR-1 brightness temperatures live in roughly 190-320 K, and water vapour is
narrow and always cold. Anything outside that is a format problem to solve
today, not in week six.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

from .grids import india_area
from .sentinels import detect_pileups

# Native resolutions, straight from the satpy reader spec.
INSAT_RESOLUTION_M = {"VIS": 1000, "SWIR": 1000, "MIR": 4000,
                      "TIR1": 4000, "TIR2": 4000, "WV": 8000}

# Brightness-temperature checks are on ROBUST PERCENTILES, not min/max.
#
# Why not min/max: the absolute extremes of a real scan are contaminated two
# ways, both measured on 3DIMG_25AUG2019_2330_L1B_STD_V01R00.
#
#   * Scattered bad pixels at the warm end. TIR1 p99.9 = 301.1 K and
#     p99.99 = 302.1 K, then a 28 K jump to a 330.6 K maximum held by
#     ~0% of pixels, 62% of them isolated singletons. There is no warm
#     region -- it is noise.
#   * LUT saturation at the cold end. The count->temperature LUT is
#     inverted and CLAMPS: LUT[-1] == LUT[-2] == 180.09 K for TIR1. So
#     0.33% of finite pixels read exactly 180.09 K, and WV independently
#     clamps at 179.89 K. The two channels sharing a floor is the clamp,
#     not physics -- WV sees only the upper troposphere and its p0.1 is
#     196.2 K, well above TIR1's.
#
# Widening the bounds to admit those extremes would fit the check to the
# data and defeat its purpose. Bounding p1 and p99.9 instead keeps the
# check meaningful while being immune to both artefacts.
#
# (percentile, (min_allowed, max_allowed)) in kelvin.
EXPECTED_BT_PERCENTILES = {
    "TIR1": {1.0: (185.0, 245.0), 99.9: (285.0, 315.0)},
    "TIR2": {1.0: (185.0, 245.0), 99.9: (285.0, 315.0)},
    "MIR": {1.0: (185.0, 250.0), 99.9: (285.0, 325.0)},
    "WV": {1.0: (200.0, 235.0), 99.9: (255.0, 280.0)},
}
BT_CHANNELS = tuple(EXPECTED_BT_PERCENTILES)

# Fraction of finite pixels allowed to sit exactly at the observed extreme
# before it is reported as saturation/fill contamination rather than signal.
MAX_SATURATED_FRAC = 0.02

# Below this many finite pixels the p1/p99.9 checks are sampling noise rather
# than physics -- the same effect measured for cross-satellite comparison
# (two identical distributions differ at p99.9 by 1.31 K at n=100k, 0.10 K at
# n=7.9M). A full TIR scan is ~5.8M valid pixels so ingest_scan is unaffected;
# a CROP may not be, and the check says so rather than passing or failing on
# noise.
MIN_RELIABLE_N = 500_000

# The off-disk fill count in the INSAT count->temperature LUT. Maps to the
# LUT's clamped final entry, so it must be masked BEFORE any statistic.
INSAT_FILL_COUNT = 1023

# Reflective channels. Meaningless at night, so they are reported N/A rather
# than FAIL when the sun is down -- see solar_elevation() below.
REFLECTIVE_CHANNELS = ("VIS", "SWIR")

# Sun elevation (degrees) below which reflective channels carry no signal.
# Civil twilight is -6; +5 leaves margin for the long slant paths near the
# terminator where reflectance is real but near-zero.
DAYLIGHT_MIN_ELEVATION = 5.0


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str
    skipped: bool = False        # not applicable (e.g. reflective at night)

    def __str__(self) -> str:
        tag = "N/A " if self.skipped else ("PASS" if self.passed else "FAIL")
        return f"[{tag}] {self.name}: {self.detail}"


@dataclass
class ScanCheck:
    results: list = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        """Skipped checks are not failures."""
        return all(r.passed for r in self.results if not r.skipped)

    def add(self, name, passed, detail, skipped=False):
        self.results.append(CheckResult(name, passed, detail, skipped))

    def report(self) -> str:
        head = "INSAT scan sanity checks: " + ("ALL PASS" if self.passed else "FAILURES")
        lines = [head, ""]
        if self.metadata:
            lines.append("  product identity (diff this against a current-version file):")
            for k, v in self.metadata.items():
                lines.append(f"    {k:26s} {v}")
            lines.append("")
        lines += [f"  {r}" for r in self.results]
        skipped = [r for r in self.results if r.skipped]
        if skipped:
            lines += ["", f"  {len(skipped)} check(s) N/A -- not failures."]
        return "\n".join(lines)

    def raise_if_failed(self):
        if not self.passed:
            bad = [r for r in self.results if not r.passed and not r.skipped]
            raise ValueError("INSAT scan failed sanity checks:\n" +
                             "\n".join(f"  {r}" for r in bad))


def read_metadata(path) -> dict:
    """Product identity from the file's root attributes.

    Reported on every check so a PASS can be attributed to a SPECIFIC product
    version. A 2019 V01R00 file passing says nothing about MOSDAC's current
    output; diff these fields against a recent file to find out.
    """
    import h5py

    want = ("Satellite_Name", "Sensor_Id", "Processing_Level", "Product_Type",
            "Software_Version", "HDF_Product_File_Name", "Acquisition_Date",
            "Acquisition_Time_in_GMT", "Sun_Elevation(Degrees)",
            "Imaging_Mode", "Radiometric_Calibration_Type", "Ground_Station",
            "Nominal_Central_Point_Coordinates(degrees)_Latitude_Longitude")
    out = {}
    with h5py.File(Path(path), "r") as fh:
        for k in want:
            if k not in fh.attrs:
                continue
            v = fh.attrs[k]
            if isinstance(v, bytes):
                v = v.decode(errors="replace")
            elif isinstance(v, np.ndarray):
                if v.dtype.kind == "S":
                    v = v[0].decode(errors="replace")
                elif v.size == 1:
                    v = float(v.flat[0])
                else:
                    # Multi-element attributes must NOT be collapsed to their
                    # first element. Nominal_Central_Point_Coordinates is
                    # [latitude, longitude]; taking flat[0] returned the
                    # latitude (0.0) and silently discarded the sub-satellite
                    # longitude that parallax geometry depends on.
                    v = [float(x) for x in v.ravel()]
            out[k] = v
    return out


def sub_satellite_longitude(meta: dict) -> float | None:
    """The satellite's sub-point longitude, from the file's own metadata.

    3DR is at 74E and 3DS at 82E, so zenith angle -- and therefore parallax --
    differs between them for the same ground cell. Read it rather than
    assuming: mixing two geometries into one parallax regression corrupts the
    slope, and the slope is the check that tells us the physics is right.
    """
    v = meta.get("Nominal_Central_Point_Coordinates(degrees)_Latitude_Longitude")
    if v is None:
        return None
    try:
        arr = np.asarray(v, dtype=float).ravel()
        return float(arr[1]) if arr.size >= 2 else None
    except (TypeError, ValueError):
        return None


def solar_elevation(meta: dict) -> float | None:
    """Scene solar elevation in degrees, or None if absent/implausible.

    INSAT-3DS writes a GARBAGE value here: 7.68e-76 on the 2025-08-01 2330Z
    scan, where the true elevation is roughly -12 deg (05:00 IST). It is
    denormal-small rather than out of range, so a naive float() accepts it
    and it reads as "0 degrees" -- which happened to gate correctly for a
    night scan and would have failed silently for a daytime one, letting
    reflective checks run on a scene whose illumination we do not know.

    Values within 1e-6 of zero are treated as unset. Use
    `solar_elevation_from_dataset` for a real value when the attribute is bad.
    """
    v = meta.get("Sun_Elevation(Degrees)")
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(f) or abs(f) > 90.0:
        return None
    if abs(f) < 1e-6:          # denormal garbage, not a real horizon crossing
        return None
    return f


def solar_elevation_from_dataset(path, aoi_lat: float = 23.0,
                                 aoi_lon: float = 82.0,
                                 radius_deg: float = 8.0) -> float | None:
    """Solar elevation over the AREA OF INTEREST, from the per-pixel dataset.

    Sampled near the AOI centre, NOT across the whole disk. A geostationary
    full disk always spans the terminator, so its median elevation is ~0 by
    construction -- measured at +0.80 deg for a 3DR scan whose own attribute
    reads -17.26 deg. A disk-median fallback would therefore report every
    scan as twilight and gate the reflective checks wrongly.

    What matters is whether OUR grid is illuminated, so the sample is taken
    within `radius_deg` of the AOI centre (default: central India).
    """
    import h5py

    with h5py.File(Path(path), "r") as fh:
        if "Sun_Elevation" not in fh or "Latitude" not in fh:
            return None
        d = fh["Sun_Elevation"]
        a = np.asarray(d[0] if d.ndim == 3 else d[:], dtype=np.float64)
        sf = d.attrs.get("scale_factor")
        if sf is not None:
            a = a * float(np.asarray(sf).flat[0])
        fill = d.attrs.get("_FillValue")
        if fill is not None:
            a = np.where(a == float(np.asarray(fill).flat[0]) *
                         (float(np.asarray(sf).flat[0]) if sf is not None else 1.0),
                         np.nan, a)
        lat = np.asarray(fh["Latitude"][:], dtype=np.float64)
        lon = np.asarray(fh["Longitude"][:], dtype=np.float64)
        for arr, key in ((lat, "Latitude"), (lon, "Longitude")):
            k = fh[key].attrs.get("scale_factor")
            if k is not None:
                arr *= float(np.asarray(k).flat[0])

    if a.shape != lat.shape:
        return None
    near = (np.abs(lat - aoi_lat) <= radius_deg) & (np.abs(lon - aoi_lon) <= radius_deg)
    v = a[near & np.isfinite(a)]
    return float(np.median(v)) if v.size else None


def is_daylight(meta: dict, min_elevation: float = DAYLIGHT_MIN_ELEVATION) -> bool:
    el = solar_elevation(meta)
    return True if el is None else el >= min_elevation


def check_physics(arrays: dict, min_valid_frac: float = 0.05,
                  metadata: dict | None = None) -> ScanCheck:
    """Validate decoded brightness temperatures before anything downstream.

    `arrays` maps channel name -> 2-D array in kelvin (or reflectance for
    VIS/SWIR). Pass `metadata` from read_metadata() so reflective channels can
    be skipped at night and the product version is recorded alongside the
    result.
    """
    meta = metadata or {}
    chk = ScanCheck(metadata=meta)
    day = is_daylight(meta)
    el = solar_elevation(meta)
    if el is not None:
        chk.add("illumination", True,
                f"sun elevation {el:+.1f} deg -- "
                + ("daylight" if day else "night/twilight, reflective channels N/A"))
    for name, arr in arrays.items():
        a = np.asarray(arr, dtype=np.float64)
        finite = np.isfinite(a)
        frac = float(finite.mean())

        # A dark VIS channel at 05:00 IST is correct behaviour, not a fault.
        if name in REFLECTIVE_CHANNELS and not day:
            chk.add(f"{name} (reflective)", True,
                    f"skipped -- sun elevation {el:+.1f} deg, no illumination",
                    skipped=True)
            continue

        chk.add(f"{name} coverage", frac >= min_valid_frac,
                f"{100 * frac:.1f}% finite pixels")

        n_finite = int(finite.sum())
        if n_finite < MIN_RELIABLE_N:
            chk.add(f"{name} sample size", True,
                    f"only {n_finite:,} finite pixels (< {MIN_RELIABLE_N:,}) -- "
                    f"p1/p99.9 here carry sampling noise of order 1 K; treat the "
                    f"percentile verdicts as indicative. Check full scans, not crops.",
                    skipped=True)
        if not finite.any():
            chk.add(f"{name} range", False, "no finite pixels to check")
            continue

        v = a[finite]
        if name in EXPECTED_BT_PERCENTILES:
            for q, (elo, ehi) in EXPECTED_BT_PERCENTILES[name].items():
                got = float(np.percentile(v, q))
                ok = elo <= got <= ehi
                chk.add(f"{name} p{q:g}", ok,
                        f"{got:.1f} K (expected {elo:.0f}-{ehi:.0f} K)"
                        + ("" if ok else "  <-- wrong units, calibration, or format drift"))

            # Report the raw extremes as context, never as a pass/fail.
            chk.add(f"{name} extremes", True,
                    f"min {v.min():.1f} / max {v.max():.1f} K (context only -- "
                    f"contaminated by LUT clamping and bad pixels)", skipped=True)

            # LUT clamping / fill contamination at either end.
            for end, val in (("floor", v.min()), ("ceiling", v.max())):
                frac = float((v == val).mean())
                chk.add(f"{name} {end} saturation", frac <= MAX_SATURATED_FRAC,
                        f"{100 * frac:.3f}% of finite pixels sit exactly at "
                        f"{val:.2f} K"
                        + ("" if frac <= MAX_SATURATED_FRAC
                           else "  <-- LUT clamp or unmasked fill value"))

        # Standing sentinel audit. Three fill/clamp values on this project
        # have decoded to plausible physical extremes without raising
        # (VIL byte-255, the int16 sentinel, the count->K LUT clamp), so this
        # runs on every channel of every source rather than being remembered.
        pu = detect_pileups(a, min_fraction=0.001)
        bad = [p for p in pu.pileups if p.suspicion == "high"]
        chk.add(f"{name} value pile-up", not bad,
                (f"{len(bad)} suspicious value(s): "
                 + "; ".join(f"{p.value:g} at {100 * p.fraction:.2f}%" for p in bad)
                 + "  <-- check the product docs; a fill decoded as a physical "
                   "extreme does not raise") if bad else "no isolated spikes")

    # The window/absorption contrast, on percentiles for the same reason.
    if "TIR1" in arrays and "WV" in arrays:
        t, w = np.asarray(arrays["TIR1"]), np.asarray(arrays["WV"])
        tf, wf = t[np.isfinite(t)], w[np.isfinite(w)]
        if tf.size and wf.size:
            tspan = float(np.percentile(tf, 99) - np.percentile(tf, 1))
            wspan = float(np.percentile(wf, 99) - np.percentile(wf, 1))
            chk.add("WV narrower than TIR1", wspan < tspan,
                    f"WV p1-p99 span {wspan:.1f} K vs TIR1 {tspan:.1f} K")
            tmed, wmed = float(np.median(tf)), float(np.median(wf))
            chk.add("WV colder than TIR1", wmed < tmed,
                    f"WV median {wmed:.1f} K vs TIR1 median {tmed:.1f} K")
    return chk


def read_scan_native(path, channels: Sequence[str] = ("TIR1", "WV")) -> dict:
    """Decode INSAT L1B directly from the HDF5, bypassing satpy.

    Needed because satpy 0.60's `insat3d_img_l1b_h5` reader FAILS on
    INSAT-3DS:

        KeyError: "No variable named 'Longitude_WV'"

    On INSAT-3D and 3DR the water-vapour channel is half-resolution
    (1408x1402, 8 km) and carries its OWN coarse geolocation arrays. On
    INSAT-3DS the WV channel was upgraded to full 4 km resolution
    (2816x2805), so it shares the main Latitude/Longitude and the separate
    `*_WV` arrays no longer exist. The reader assumes they do.

    This path resolves geolocation per channel: `*_WV` when present, the
    main arrays otherwise. It returns brightness temperature in kelvin with
    the off-disk fill masked to NaN.
    """
    import h5py

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"no INSAT file at {p}")

    out = {}
    with h5py.File(p, "r") as fh:
        for ch in channels:
            img, lut = f"IMG_{ch}", f"IMG_{ch}_TEMP"
            if img not in fh:
                raise KeyError(f"{p.name} has no {img}; found {sorted(fh.keys())[:12]}")
            if lut not in fh:
                raise KeyError(f"{p.name} has no {lut} -- cannot convert counts to "
                               f"brightness temperature without the lookup table")
            counts = np.asarray(fh[img][0], dtype=np.int32)
            table = np.asarray(fh[lut][:], dtype=np.float64)
            bt = table[np.clip(counts, 0, table.size - 1)]
            bt[counts == INSAT_FILL_COUNT] = np.nan

            lat_key = f"Latitude_{ch}" if f"Latitude_{ch}" in fh else "Latitude"
            lon_key = f"Longitude_{ch}" if f"Longitude_{ch}" in fh else "Longitude"
            lat = np.asarray(fh[lat_key][:], dtype=np.float64)
            lon = np.asarray(fh[lon_key][:], dtype=np.float64)
            for arr, key in ((lat, lat_key), (lon, lon_key)):
                sf = fh[key].attrs.get("scale_factor")
                if sf is not None:
                    arr *= float(np.asarray(sf).flat[0])
            out[ch] = {"bt": bt, "lat": lat, "lon": lon,
                       "geo_source": lat_key}
    return out


def read_scan(path, channels: Sequence[str] = ("TIR1", "WV"),
              calibration: str = "brightness_temperature"):
    """Open one INSAT L1B file with satpy. Returns a satpy Scene."""
    from satpy import Scene

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"no INSAT file at {p}")
    scn = Scene(filenames=[str(p)], reader="insat3d_img_l1b_h5")

    available = set(scn.available_dataset_names())
    missing = [c for c in channels if c not in available]
    if missing:
        raise KeyError(
            f"reader exposes {sorted(available)} but {missing} were requested. "
            f"If the product format has drifted, this is where it shows up.")
    scn.load(list(channels), calibration=calibration)
    return scn


def resample_to_grid(scene, channels: Sequence[str] = ("TIR1", "WV"),
                     resampler: str = "nearest",
                     radius_of_influence: float = 12000.0) -> dict:
    """Resample loaded channels onto the common India 4 km grid.

    Nearest by default with a 12 km search radius -- generous enough to fill
    the 4 km grid from the 8 km water-vapour channel without inventing
    structure, which is the same honesty as the nearest-upsampling used on
    the SEVIR side.
    """
    resampled = scene.resample(india_area(), resampler=resampler,
                               radius_of_influence=radius_of_influence)
    return {c: np.asarray(resampled[c].values, dtype=np.float32) for c in channels}


def ingest_scan(path, channels: Sequence[str] = ("TIR1", "WV"),
                strict: bool = True) -> tuple[dict, ScanCheck]:
    """Read, sanity-check at native resolution, then resample.

    Checks run BEFORE resampling: resampling a mis-decoded field produces a
    smooth, plausible-looking array and makes the original problem harder to
    see.
    """
    meta = read_metadata(path)
    scn, native_geo = None, None
    try:
        scn = read_scan(path, channels)
        native = {c: np.asarray(scn[c].values, dtype=np.float64) for c in channels}
        reader_used = "satpy"
    except Exception as e:
        # satpy fails on INSAT-3DS (missing Longitude_WV). Fall back rather
        # than losing the current operational satellite entirely.
        native_geo = read_scan_native(path, channels)
        native = {c: v["bt"] for c, v in native_geo.items()}
        reader_used = f"native (satpy failed: {type(e).__name__})"
    meta = dict(meta) | {"_reader": reader_used}
    if solar_elevation(meta) is None:
        # attribute unusable (INSAT-3DS writes denormal garbage) -- recover
        # illumination from the per-pixel dataset rather than guessing
        el = solar_elevation_from_dataset(path)
        if el is not None:
            meta["Sun_Elevation(Degrees)"] = el
            meta["_sun_elevation_source"] = "Sun_Elevation dataset (attribute unusable)"
    chk = check_physics(native, metadata=meta)
    if strict:
        chk.raise_if_failed()
    if scn is not None:
        return resample_to_grid(scn, channels), chk
    return resample_native_to_grid(native_geo), chk


def resample_native_to_grid(native_geo: dict,
                            radius_of_influence: float = 12000.0) -> dict:
    """Resample the native-reader output onto the common India grid.

    Each channel carries its own geolocation, which matters: on 3D/3DR the
    water-vapour channel is on a different (coarser) grid from TIR.
    """
    from pyresample.geometry import SwathDefinition
    from pyresample.kd_tree import resample_nearest

    area = india_area()
    out = {}
    for ch, v in native_geo.items():
        lat, lon, bt = v["lat"], v["lon"], v["bt"]
        good = (np.isfinite(lat) & np.isfinite(lon)
                & (np.abs(lat) <= 90) & (np.abs(lon) <= 180))
        swath = SwathDefinition(lons=np.where(good, lon, np.nan),
                                lats=np.where(good, lat, np.nan))
        filled = np.where(np.isfinite(bt), bt, -999.0)
        r = resample_nearest(swath, filled, area,
                             radius_of_influence=radius_of_influence,
                             fill_value=-999.0)
        out[ch] = np.where(r < 0, np.nan, r).astype(np.float32)
    return out


def fetch_scan(*_args, **_kw):
    """Not implemented -- deliberately.

    MOSDAC serves INSAT products through an order/download portal whose
    endpoint and auth flow are not stable enough to hard-code blind. Guessing
    a URL here would produce a client that fails confusingly against the real
    service.

    Download one scan by hand from https://mosdac.gov.in (credentials in
    .env), then point `ingest_scan` at the local file. Once the actual
    request shape is confirmed against the live portal, implement it here
    using nowcast_data.credentials.get("MOSDAC_USERNAME"/"MOSDAC_PASSWORD").
    """
    raise NotImplementedError(
        "MOSDAC fetch is not implemented -- download a scan manually from "
        "https://mosdac.gov.in and pass the local path to ingest_scan(). "
        "See the docstring for why this is not guessed.")
