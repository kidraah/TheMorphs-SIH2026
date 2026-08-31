"""The frozen config. Every field must move the fingerprint, or it is not frozen.

Bulk ingest transfers ~3.4 TB and discards the raw files. A field that is
recorded but not hashed would let two different caches share an id, and the
manifest check would pass while the bytes differed -- the failure this whole
mechanism exists to prevent, one level down.
"""
import dataclasses

import pytest

from nowcast_train.insat_cache import ALL_CHANNELS, InsatCacheConfig


def test_every_field_changes_the_fingerprint():
    base = InsatCacheConfig()
    fp = base.fingerprint()
    alt = {"grid_shape": (900, 900), "target_km": 8.0, "projection": "merc",
           "proj_lat_0": 20.0, "proj_lon_0": 80.0, "resample": "bilinear",
           "radius_of_influence_m": 20000.0, "channels": ("TIR1", "WV"),
           "store_dtype": "float32", "cadence_min": 15.0,
           "per_scan_fields": ("scan_time_utc",),
           "static_channels": ("hand_m",),
           "static_source_versions": ("MERIT_Hydro_v1.0.2",),
           "antecedent_channel": False, "antecedent_days": 3,
           "antecedent_excludes_current_day": False,
           "antecedent_source": "IMERG-Late-V07",
           "antecedent_lead_in_days": 7, "fetch_mode": "whole",
           "geolocation_fetched_once": False,
           "label_source": "IMERG-Late-V07", "label_km": 4.0,
           "extreme_percentile": 99.5, "label_climatology": "global",
           "nan_policy": "FILL_ZERO", "store_finite_mask": False,
           "version": 3, "decode_version": 99, "notes": "x"}
    fields = {f.name for f in dataclasses.fields(base)}
    assert fields == set(alt), f"untested fields: {fields ^ set(alt)}"
    for name, value in alt.items():
        kw = {name: value}
        if name == "antecedent_days":
            kw["antecedent_lead_in_days"] = 7
        assert InsatCacheConfig(**kw).fingerprint() != fp, name


def test_fingerprint_is_stable_across_processes():
    """hash() is randomised per process; this must not be."""
    assert InsatCacheConfig().fingerprint() == InsatCacheConfig().fingerprint()
    assert InsatCacheConfig().fingerprint() == "90e78e734db78158", (
        "the default config changed -- that is a re-ingest, so it must be "
        "deliberate. Update this value in the same commit that changes it.")


def test_range_support_inverts_the_channel_decision():
    """When transfer was fixed at 448 MB, keeping all six channels was free
    insurance. MOSDAC honours Range (verified byte-identical, 14x saving), so
    transfer is no longer already paid and VIS/SWIR cost 23x."""
    from nowcast_train.insat_cache import FETCH_CHANNELS
    c = InsatCacheConfig()
    assert c.channels == FETCH_CHANNELS == ("TIR1", "TIR2", "MIR", "WV")
    assert "VIS" not in c.channels and "SWIR" not in c.channels
    assert c.fetch_mode == "ranged" and c.geolocation_fetched_once


def test_unknown_channel_is_rejected():
    with pytest.raises(ValueError, match="unknown channels"):
        InsatCacheConfig(channels=("TIR1", "TIR3"))


def test_antecedent_lead_in_must_cover_the_window():
    """Without the lead-in the antecedent channel is nan for the first days
    of every event -- which is the start of the storm."""
    with pytest.raises(ValueError, match="lead-in"):
        InsatCacheConfig(antecedent_days=5, antecedent_lead_in_days=2)
    InsatCacheConfig(antecedent_days=5, antecedent_lead_in_days=5)


def test_per_scan_metadata_covers_what_cannot_be_recovered_later():
    """Each of these has already caused, or would cause, a wrong number if
    assumed rather than stored: the 3DS solar-elevation attribute is denormal
    garbage, sub-satellite longitude differs between satellites and drives
    parallax, and the reader fallback changes the decode path."""
    f = set(InsatCacheConfig().per_scan_fields)
    for needed in ("sub_satellite_longitude", "solar_elevation", "satellite",
                   "reader", "scan_time_utc"):
        assert needed in f, needed


def test_archive_size_is_what_the_plan_assumes():
    c = InsatCacheConfig()
    gb = c.archive_gb(19200)
    assert 170 < gb < 230, gb          # ~197 GB, vs 8,200 GB of whole files


def test_decode_behaviour_is_fingerprinted():
    """The LUT-clamp fix turns 0.3-1.1% of cells in three of four channels
    from a number into NaN -- a change to every cached byte -- while every
    other fingerprinted field stayed identical. The freeze was not stale, it
    was BLIND, which is worse: a stale freeze announces itself."""
    a = InsatCacheConfig()
    b = InsatCacheConfig(decode_version=a.decode_version + 1)
    assert a.fingerprint() != b.fingerprint()
    assert a.decode_version >= 2, "1 was the half-fixed clamp"
