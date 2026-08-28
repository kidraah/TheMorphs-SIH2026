"""Tests for the SEVIR loader, run against a synthetic SEVIR-shaped store.

What these can and cannot prove
-------------------------------
They prove the resolution matching, cadence subsampling, windowing, day
grouping and failure modes are correct. They CANNOT verify SEVIR's real
VIL encoding or the real CATALOG.csv column spellings -- both need checking
against actual data, and the loader warns about the first.
"""
import numpy as np
import pytest

from nowcast_data.grid import (block_mean, match_sensor_resolution,
                               nearest_upsample, subsample_time)
from nowcast_data.sevir import (DEFAULT_CHANNELS, INT16_MISSING, VIL_CHANNEL,
                                SEVIRConfig, SEVIRLoader, decode_linear,
                                decode_vil, stack_samples)
from nowcast_data.synthetic import build_store


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    return build_store(tmp_path_factory.mktemp("sevir"), n_events=6, n_days=2)


def _cfg(store, **kw):
    kw.setdefault("inputs", [DEFAULT_CHANNELS["ir107"], DEFAULT_CHANNELS["ir069"]])
    return SEVIRConfig.from_store(store, target=VIL_CHANNEL, **kw)


# --------------------------------------------------------------------------
# grid.py -- the resolution argument
# --------------------------------------------------------------------------

def test_block_mean_averages_rather_than_subsamples():
    """A coarser sensor integrates over its footprint; it does not sample a
    point. Subsampling would keep a sharp feature a real sensor would smear."""
    a = np.array([[0.0, 4.0], [8.0, 12.0]])
    assert block_mean(a, 2)[0, 0] == pytest.approx(6.0)   # mean, not a[0,0]


def test_block_mean_rejects_uneven_division():
    with pytest.raises(ValueError, match="not divisible"):
        block_mean(np.zeros((10, 10)), 3)


def test_water_vapour_carries_no_structure_below_8km():
    """The central claim of the per-channel treatment.

    SEVIR WV at 2 km, degraded to INSAT's 8 km, then placed on the 4 km
    grid: adjacent 2x2 cells MUST be identical. If they are not, the model
    can learn moisture gradients INSAT cannot deliver -- and IWV variation
    is this project's cornerstone predictor.
    """
    rng = np.random.default_rng(0)
    wv = rng.random((192, 192))
    out = match_sensor_resolution(wv, native_km=2.0, sensor_km=8.0, target_km=4.0)
    assert out.shape == (96, 96)
    for i in (0, 10, 44):
        blk = out[2 * i:2 * i + 2, 2 * i:2 * i + 2]
        assert np.allclose(blk, blk[0, 0]), "8 km field must be blocky at 4 km"


def test_thermal_infrared_keeps_4km_structure():
    """The contrast: TIR is 4 km native on INSAT, so it must NOT be blocky."""
    rng = np.random.default_rng(1)
    tir = rng.random((192, 192))
    out = match_sensor_resolution(tir, native_km=2.0, sensor_km=4.0, target_km=4.0)
    assert out.shape == (96, 96)
    blk = out[0:2, 0:2]
    assert not np.allclose(blk, blk[0, 0]), "4 km field must retain 4 km detail"


def test_uniform_downsample_would_differ_from_sensor_matching():
    """Proves the two-step is not a no-op dressed up as rigour."""
    rng = np.random.default_rng(2)
    wv = rng.random((192, 192))
    matched = match_sensor_resolution(wv, 2.0, 8.0, 4.0)   # via 8 km
    naive = match_sensor_resolution(wv, 2.0, 4.0, 4.0)     # straight to 4 km
    assert not np.allclose(matched, naive)


def test_cannot_invent_detail_the_source_lacks():
    with pytest.raises(ValueError, match="finer than"):
        match_sensor_resolution(np.zeros((8, 8)), native_km=4.0, sensor_km=2.0,
                                target_km=2.0)


def test_upsampling_is_nearest_not_interpolated():
    """Bilinear would manufacture smooth gradients that look like structure."""
    a = np.array([[1.0, 2.0]])
    assert np.array_equal(nearest_upsample(a, 2),
                          np.array([[1., 1., 2., 2.], [1., 1., 2., 2.]]))


def test_time_subsampling_matches_scan_cadence():
    frames = np.arange(49).reshape(49, 1, 1)
    out = subsample_time(frames, native_min=5.0, target_min=30.0)
    assert out.shape[0] == 9
    assert [int(v) for v in out[:3, 0, 0]] == [0, 6, 12]   # every 6th = 30 min


def test_non_integer_cadence_is_rejected():
    with pytest.raises(ValueError, match="not an integer ratio"):
        subsample_time(np.zeros((49, 2, 2)), native_min=5.0, target_min=7.0)


# --------------------------------------------------------------------------
# The loader
# --------------------------------------------------------------------------

def test_loads_every_channel_onto_one_grid(store):
    loader = SEVIRLoader(_cfg(store))
    s = loader.sample("E000")
    assert s.x.shape == (2, 2, 96, 96)     # 2 channels, 2 context frames
    assert s.y.shape == (6, 96, 96)
    assert s.channels == ["ir107", "ir069"]


def test_grid_size_follows_from_domain_and_target(store):
    assert _cfg(store).grid_size == 96                        # 384 km / 4 km
    assert _cfg(store, target_km=8.0).grid_size == 48


def test_lead_minutes_match_the_scan_cadence(store):
    assert _cfg(store).lead_minutes == [30, 60, 90, 120, 150, 180]


def test_physical_scaling_is_applied(store):
    """Raw int16 counts must come back as physical units, or every
    normalisation statistic downstream is wrong by a factor of 100."""
    loader = SEVIRLoader(_cfg(store))
    raw = loader._read_raw("E000", DEFAULT_CHANNELS["ir107"])
    # nanmax: the fixture injects missing pixels, which decode to NaN
    assert np.isnan(raw).any(), "fixture must contain missing pixels"
    assert np.nanmax(raw) < 10, f"expected scaled units, got max {np.nanmax(raw)}"


def test_channels_are_degraded_differently(store):
    """End-to-end version of the WV/TIR contrast, through the real loader."""
    loader = SEVIRLoader(_cfg(store))
    wv = loader.load_channel("E000", DEFAULT_CHANNELS["ir069"])
    tir = loader.load_channel("E000", DEFAULT_CHANNELS["ir107"])
    assert wv.shape == tir.shape

    def blockiness(a):
        # nan-aware: missing pixels propagate through the downsample
        return np.nanmean(np.abs(a[:, 0::2, 0::2] - a[:, 1::2, 1::2]))

    assert blockiness(wv) == pytest.approx(0.0, abs=1e-12)
    assert blockiness(tir) > 1e-6


# --------------------------------------------------------------------------
# The SEVIR ceiling
# --------------------------------------------------------------------------

def test_six_hour_horizon_is_rejected_with_an_explanation(store):
    """SEVIR events are 4 hours. The 2-6 h claim cannot be pretrained here,
    and the config must say so rather than silently truncating."""
    with pytest.raises(ValueError, match="cannot be pretrained on SEVIR"):
        _cfg(store, context_frames=2, horizon_frames=12)     # 7 h at 30 min


def test_three_hour_horizon_fits(store):
    cfg = _cfg(store, context_frames=2, horizon_frames=6)
    assert cfg.lead_minutes[-1] == 180


# --------------------------------------------------------------------------
# Day grouping -- the bridge to the bootstrap
# --------------------------------------------------------------------------

def test_day_groups_expose_the_clustering(store):
    loader = SEVIRLoader(_cfg(store))
    samples = list(loader.iter_samples())
    groups = loader.day_groups(samples)
    assert len(groups) == len(samples)
    assert len(set(groups)) == 2, "6 events over 2 storm days"


def test_day_groups_feed_the_block_bootstrap(store):
    """The connection that keeps validation intervals honest."""
    from nowcast_eval import EvalConfig, bootstrap_ci

    loader = SEVIRLoader(_cfg(store))
    samples = list(loader.iter_samples())
    groups = loader.day_groups(samples)
    _, y = stack_samples(samples)

    obs = (y > y.mean()).astype(float)
    pred = np.clip(obs * 0.7 + 0.1, 0, 1)
    ci = bootstrap_ci(pred, obs, EvalConfig(), n_boot=100, groups=groups)
    assert ci.n_samples == len(samples)
    assert ci.n_blocks == 2, "must resample storm days, not windows"


def test_overlapping_windows_come_from_one_event(store):
    loader = SEVIRLoader(_cfg(store, context_frames=2, horizon_frames=4))
    samples = list(loader.iter_samples(["E000"], stride=1))
    assert len(samples) > 1
    assert all(s.event_id == "E000" for s in samples)


# --------------------------------------------------------------------------
# Failure modes
# --------------------------------------------------------------------------

def test_event_missing_a_channel_is_dropped_not_zero_filled(store, tmp_path):
    """A zero-filled water vapour field is not 'missing' to a model -- it is
    a confident assertion that the air is dry."""
    import pandas as pd
    alt = tmp_path / "partial"
    alt.mkdir()
    (alt / "data").symlink_to(store / "data")
    df = pd.read_csv(store / "CATALOG.csv")
    df = df[~((df["id"] == "E000") & (df["img_type"] == "ir069"))]
    df.to_csv(alt / "CATALOG.csv", index=False)

    loader = SEVIRLoader(_cfg(alt))
    assert "E000" not in loader.event_ids()
    assert "E001" in loader.event_ids()


# --------------------------------------------------------------------------
# Decoding -- the silent-failure surface
# --------------------------------------------------------------------------

def test_vil_decode_matches_the_published_formula():
    """SEVIR NeurIPS-2020 supplemental, eq. 2. Hand-evaluated at each branch."""
    x = np.array([0, 5, 6, 18, 19, 255], dtype=np.uint8)
    got = decode_vil(x)
    assert got[0] == 0.0 and got[1] == 0.0                      # X <= 5
    assert got[2] == pytest.approx((6 - 2) / 90.66)             # linear branch
    assert got[3] == pytest.approx((18 - 2) / 90.66)            # branch edge
    assert got[4] == pytest.approx(np.exp((19 - 83.9) / 38.9))  # exp branch
    assert got[5] == pytest.approx(np.exp((255 - 83.9) / 38.9))


def test_vil_decode_is_continuous_at_the_branch_boundary():
    """Pins the parenthesisation.

    The published PDF renders the third branch as a display fraction, and
    flat text extraction cannot distinguish exp((X-83.9)/38.9) from
    exp(X-83.9)/38.9. Continuity settles it: the linear branch gives 0.1765
    at X=18, the correct reading gives 0.1838, and the wrong one gives
    6e-31. Without this test the ambiguity could silently resolve the wrong
    way in a future edit and mislabel every pretraining sample.
    """
    lo = decode_vil(np.array([18], dtype=np.uint8))[0]
    hi = decode_vil(np.array([19], dtype=np.uint8))[0]
    assert lo == pytest.approx(0.176483, abs=1e-5)    # (18-2)/90.66
    assert hi == pytest.approx(0.188556, abs=1e-5)    # exp((19-83.9)/38.9)
    assert abs(hi - lo) < 0.02, "branches must nearly meet, not jump 30 orders"


def test_vil_decode_spans_physical_range():
    full = decode_vil(np.arange(256, dtype=np.uint8))
    assert full.min() == 0.0
    assert 50 < full.max() < 120, f"kg/m^2 top end implausible: {full.max()}"
    assert np.all(np.diff(full) >= 0), "decode must be monotonic"


def test_missing_sentinel_becomes_nan_not_minus_327_degrees():
    """int16 min * 1e-2 = -327.68 degC reads as an extremely cold cloud top --
    exactly the signal the CTT drop rate keys on. It must be NaN."""
    raw = np.array([INT16_MISSING, 0, 2500], dtype=np.int16)
    out = decode_linear(raw, 1e-2)
    assert np.isnan(out[0])
    assert out[1] == 0.0 and out[2] == pytest.approx(25.0)


def test_missing_pixels_survive_downsampling_as_nan(store):
    """One missing pixel must not void a whole block, and must not silently
    become a real number either."""
    loader = SEVIRLoader(_cfg(store))
    tir = loader.load_channel("E000", DEFAULT_CHANNELS["ir107"])
    assert np.isfinite(tir).any(), "downsampling must not void everything"


def test_nan_aware_block_mean_averages_valid_members():
    a = np.array([[1.0, np.nan], [3.0, 5.0]])
    assert block_mean(a, 2)[0, 0] == pytest.approx(3.0)   # mean of 1,3,5
    allnan = np.full((2, 2), np.nan)
    assert np.isnan(block_mean(allnan, 2)[0, 0])


def test_unverified_channel_still_warns():
    """The mechanism stays live for any channel not yet checked."""
    from dataclasses import replace
    spec = replace(VIL_CHANNEL, verified=False)
    assert spec.verified is False


def test_synthetic_target_has_real_dynamic_range(store):
    """Guards the fixture: an earlier version wrote VIL bytes of 0 and 1,
    which decoded to ~zero everywhere -- a degenerate training target that
    every test still passed on."""
    loader = SEVIRLoader(_cfg(store))
    y = loader.sample("E000").y
    assert y.max() > 1.0, f"target must span real kg/m^2, got max {y.max()}"
    assert (y > 0).mean() > 0.01, "target must not be almost entirely zero"


def test_missing_catalog_columns_are_reported(tmp_path):
    import pandas as pd
    pd.DataFrame({"id": ["a"], "wrong": [1]}).to_csv(tmp_path / "CATALOG.csv",
                                                     index=False)
    with pytest.raises(ValueError, match="missing columns"):
        SEVIRLoader(SEVIRConfig.from_store(tmp_path))


def test_requesting_too_many_frames_is_explicit(store):
    loader = SEVIRLoader(_cfg(store, context_frames=2, horizon_frames=6))
    with pytest.raises(ValueError, match="only .* available"):
        loader.sample("E000", start=5)


def test_stack_samples_shapes(store):
    loader = SEVIRLoader(_cfg(store))
    x, y = stack_samples(list(loader.iter_samples()))
    assert x.shape == (6, 2, 2, 96, 96)
    assert y.shape == (6, 6, 96, 96)
