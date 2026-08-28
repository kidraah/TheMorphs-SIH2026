"""Smoke tests against the REAL 229 GB SEVIR download.

Skipped automatically when the store is absent, so CI and a fresh clone stay
green. Their job is to catch the synthetic fixture drifting away from the
real distribution -- which it already had once: the synthetic store used a
flat data/ directory while the real one nests data/<img_type>/<year>/ with
`file_name` relative to data/. A test suite that only ever sees the
synthetic layout cannot notice that.
"""
from pathlib import Path

import numpy as np
import pytest

from nowcast_data.sevir import DEFAULT_CHANNELS, VIL_CHANNEL, SEVIRConfig, SEVIRLoader

STORE = Path(__file__).resolve().parents[1] / "data" / "sevir"
pytestmark = pytest.mark.skipif(
    not (STORE / "CATALOG.csv").exists(),
    reason="real SEVIR store not present (run scripts/download_sevir.sh)")


@pytest.fixture(scope="module")
def loader():
    return SEVIRLoader(SEVIRConfig.from_store(
        STORE, inputs=[DEFAULT_CHANNELS["ir107"], DEFAULT_CHANNELS["ir069"]],
        target=VIL_CHANNEL, context_frames=2, horizon_frames=6))


def test_catalog_has_the_columns_the_loader_needs(loader):
    for col in ("id", "file_name", "file_index", "img_type", "time_utc",
                "pct_missing"):
        assert col in loader.catalog.df.columns


def test_finds_events_with_all_three_channels(loader):
    ids = loader.event_ids()
    assert len(ids) > 10_000, f"expected ~12.9k usable events, got {len(ids)}"


def test_real_event_loads_at_the_target_grid(loader):
    s = loader.sample(loader.event_ids()[0])
    assert s.x.shape == (2, 2, 96, 96)
    assert s.y.shape == (6, 96, 96)
    assert s.day.count("-") == 2


def test_decoded_values_are_physically_plausible(loader):
    """Guards the decode end to end. A wrong scaling factor or a mishandled
    missing sentinel shows up here as absurd physics, not as an exception."""
    s = loader.sample(loader.event_ids()[0])
    tir, wv = s.x[0], s.x[1]

    # brightness temperatures in degC: cold cloud tops, never below ~-100
    assert -100 < np.nanmin(tir) < 0, f"ir107 min {np.nanmin(tir)}"
    assert np.nanmax(tir) < 60, f"ir107 max {np.nanmax(tir)}"

    # the water vapour channel sees only the upper troposphere, so it is
    # always cold and spans a much narrower range than the window channel
    assert np.nanmax(wv) < 0, f"ir069 should be entirely cold, max {np.nanmax(wv)}"
    assert np.nanmax(wv) - np.nanmin(wv) < np.nanmax(tir) - np.nanmin(tir)

    # VIL in kg/m^2, non-negative and bounded
    assert np.nanmin(s.y) >= 0.0
    assert np.nanmax(s.y) < 120.0


def test_missing_pixels_are_nan_not_minus_327(loader):
    """Real SEVIR carries the int16 missing sentinel. Undecoded it becomes
    -327.68 degC, which reads as an extremely cold cloud top."""
    s = loader.sample(loader.event_ids()[0])
    assert np.isnan(s.x).any(), "real data should contain missing pixels"
    assert np.nanmin(s.x) > -100, "sentinel leaked through as a real value"


def test_water_vapour_is_blocky_at_8km(loader):
    """The per-channel resolution decision, verified on real data: WV must
    carry no structure below INSAT's 8 km."""
    s = loader.sample(loader.event_ids()[0])
    wv = s.x[1]
    diff = np.nanmean(np.abs(wv[:, 0::2, 0::2] - wv[:, 1::2, 1::2]))
    assert diff == pytest.approx(0.0, abs=1e-9), f"WV not 8 km-blocky: {diff}"


def test_events_cluster_heavily_by_storm_day(loader):
    """The empirical basis for day-blocking the bootstrap.

    ~12.9k events come from only ~526 storm days. Omitting `groups` would
    overstate the effective sample size by roughly 24x.
    """
    import pandas as pd
    ids = loader.event_ids()
    sub = loader.catalog.df[loader.catalog.df["id"].isin(ids)].drop_duplicates("id")
    days = pd.to_datetime(sub["time_utc"]).dt.date
    assert days.nunique() < len(ids) / 10, (
        f"{len(ids)} events over {days.nunique()} days -- day blocking is "
        f"essential and this test exists to keep that visible")
