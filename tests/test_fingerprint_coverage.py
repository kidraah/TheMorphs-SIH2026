"""Does every fingerprinted field actually reach the bytes, and vice versa?

The blind-freeze finding generalises: a freeze that does not cover the code
path producing the bytes is not a freeze. It fails in TWO directions and both
were present here.

  BLIND   a behaviour determines the cached bytes and is not hashed.
          Changing it silently produces different data under the same
          fingerprint. This is the dangerous one -- a stale freeze announces
          itself on the next load, a blind one never does.

  HOLLOW  a field is hashed and never read. Changing it produces a new
          fingerprint, a new cache directory and byte-identical contents.
          Not dangerous, but it makes the fingerprint a claim the code does
          not honour, which is how you stop trusting it.

This file pins both directions so neither can return quietly.
"""
import inspect

import numpy as np
import pytest

from nowcast_train import insat_ingest
from nowcast_train.insat_cache import InsatCacheConfig

# Fields that legitimately do not touch a granule's cached bytes, each with
# the reason. Anything not listed here MUST be read by the ingest path.
EXEMPT = {
    "cadence_min": "selects WHICH granules are ingested, not their contents",
    "static_channels": "written once as a separate static file, not per granule",
    "static_source_versions": "provenance for the static file",
    "antecedent_channel": "derived from IMERG in a later stage",
    "antecedent_days": "later stage",
    "antecedent_excludes_current_day": "later stage",
    "antecedent_source": "later stage",
    "antecedent_lead_in_days": "an ingest-WINDOW constraint, not a byte one",
    "label_source": "labels are a separate stage",
    "label_km": "labels are a separate stage",
    "extreme_percentile": "labels are a separate stage",
    "label_climatology": "labels are a separate stage",
    "fetch_mode": "how bytes are transferred, not what is stored",
    "geolocation_fetched_once": "transfer strategy",
    "version": "deliberate global invalidator",
    "notes": "free text",
    "per_scan_fields": "checked separately by test_per_scan_fields_are_stored",
    "decode_version": "covers the decode path, verified by the clamp tests",
    "projection": "honoured by REJECTION in __post_init__ -- the grid code "
                  "implements LAEA only and refuses anything else",
}


def test_no_fingerprinted_field_is_hollow():
    """Every non-exempt field must be READ somewhere in the ingest path."""
    src = inspect.getsource(insat_ingest)
    import nowcast_data.insat as insat
    src += inspect.getsource(insat)
    missing = []
    for f in InsatCacheConfig().__dataclass_fields__:
        if f in EXEMPT:
            continue
        if f"cfg.{f}" not in src and f"config.{f}" not in src:
            missing.append(f)
    assert not missing, (
        f"hashed but never read: {missing}. Changing any of these produces a "
        f"new fingerprint and byte-identical data -- the fingerprint would be "
        f"making a claim the code does not honour.")


def test_store_dtype_matches_what_is_written(tmp_path):
    """The config said uint16 while the code wrote float32, and
    bytes_per_scan() computed the 197 GB budget from uint16 -- wrong by 2x
    uncompressed. A field that contradicts the code is worse than no field."""
    cfg = InsatCacheConfig()
    arr = {c: np.full((8, 8), 250.0) for c in cfg.channels}
    out = insat_ingest._write_cache_entry(tmp_path, cfg, arr, sub_lon=74.0,
                                          checks_passed=True, meta={})
    with np.load(out) as z:
        assert z["data"].dtype == np.dtype(cfg.store_dtype), (
            f"config says {cfg.store_dtype}, wrote {z['data'].dtype}")


def test_bytes_per_scan_agrees_with_a_real_write(tmp_path):
    cfg = InsatCacheConfig()
    h, w = cfg.grid_shape
    arr = {c: np.full((h, w), 250.0) for c in cfg.channels}
    out = insat_ingest._write_cache_entry(tmp_path, cfg, arr, sub_lon=74.0,
                                          checks_passed=True, meta={})
    with np.load(out) as z:
        raw = sum(v.nbytes for k, v in z.items() if k in ("data", "finite"))
    assert abs(raw - cfg.bytes_per_scan()) / cfg.bytes_per_scan() < 0.25, (
        f"wrote {raw/1e6:.1f} MB, bytes_per_scan() says "
        f"{cfg.bytes_per_scan()/1e6:.1f} MB")


def test_nan_policy_and_finite_mask_are_honoured(tmp_path):
    cfg = InsatCacheConfig(store_finite_mask=False)
    arr = {c: np.full((8, 8), np.nan) for c in cfg.channels}
    out = insat_ingest._write_cache_entry(tmp_path, cfg, arr, sub_lon=74.0,
                                          checks_passed=True, meta={})
    with np.load(out) as z:
        assert "finite" not in z, "store_finite_mask=False must omit it"


def test_per_scan_fields_are_stored(tmp_path):
    """Each of these was chosen because it cannot be recovered later."""
    cfg = InsatCacheConfig()
    arr = {c: np.full((8, 8), 250.0) for c in cfg.channels}
    out = insat_ingest._write_cache_entry(
        tmp_path, cfg, arr, sub_lon=74.0, checks_passed=True,
        meta={"scan_time_utc": "2024-06-26T00:15:00Z", "satellite": "3RIMG",
              "reader": "satpy"})
    with np.load(out) as z:
        for k in ("sub_satellite_longitude", "scan_time_utc", "satellite",
                  "reader", "physics_check_passed"):
            assert k in z, f"{k} not stored"



def test_projection_is_honoured_by_rejection():
    """A field the code cannot deliver is the hollow direction of the
    blind-freeze failure. Refusing the value keeps it truthful."""
    with pytest.raises(ValueError, match="not implemented"):
        InsatCacheConfig(projection="merc")
    InsatCacheConfig(projection="laea")


def test_grid_fields_actually_change_the_grid():
    """grid_shape, target_km, proj_lat_0 and proj_lon_0 were hashed while
    india_area() took no arguments at all."""
    from nowcast_data.grids import india_area
    a = india_area()
    b = india_area(shape=(456, 432), resolution_m=8000.0)
    assert a.shape != b.shape
    assert india_area(lon_0=74.0).proj_dict["lon_0"] == 74.0
