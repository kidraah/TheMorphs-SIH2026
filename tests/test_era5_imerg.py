"""ERA5 thermodynamics and IMERG label construction.

The ERA5 tests that touch the network are skipped when it is unavailable;
everything else runs offline.
"""
import os

import numpy as np
import pytest

from nowcast_data import era5
from nowcast_data.imerg_labels import (CLOUDBURST_MM_HR, ClimatologyAccumulator,
                                       RainClimatology, cloudburst_labels,
                                       extreme_labels, rain_labels)


# --------------------------------------------------------------------------
# ERA5 -- derived fields, offline
# --------------------------------------------------------------------------

def test_divergence_uses_the_cos_latitude_metric():
    """Without it the zonal derivative is wrong by 1/cos(lat) -- ~1% at
    Kanyakumari and ~22% at Kashmir, a latitude-dependent bias in the
    convergence field that would look like real geography."""
    lat = np.linspace(8.0, 35.0, 40)
    lon = np.linspace(70.0, 90.0, 40)
    u = np.tile(np.linspace(0, 10, 40), (40, 1))     # pure zonal gradient
    v = np.zeros_like(u)

    d = era5._spherical_divergence(u, v, lat, lon)
    south, north = np.abs(d[0]).mean(), np.abs(d[-1]).mean()
    assert north > south, "the same du/dlon must give larger du/dx nearer the pole"
    ratio = north / south
    expected = np.cos(np.deg2rad(lat[0])) / np.cos(np.deg2rad(lat[-1]))
    assert ratio == pytest.approx(expected, rel=0.05)


def test_divergence_of_uniform_flow_is_zero():
    lat, lon = np.linspace(5, 40, 30), np.linspace(65, 100, 30)
    u, v = np.full((30, 30), 7.0), np.full((30, 30), -3.0)
    d = era5._spherical_divergence(u, v, lat, lon)
    # the metric term makes meridional flow slightly divergent on a sphere;
    # only the zonal part must vanish exactly for uniform u
    assert np.abs(d).max() < 1e-6


def test_cin_expected_range_is_positive():
    """ERA5 reports CIN as a POSITIVE magnitude. An earlier version of this
    check asserted CIN <= 0 and failed valid data."""
    lo, hi = era5.EXPECTED["cin"]
    assert lo >= 0.0 and hi > 0.0


def test_cin_is_never_filled_with_zero():
    """The most dangerous fill in the dataset: CIN is undefined where there is
    NO CONVECTIVE PARCEL. Filling that with 0 reads as 'no inhibition' --
    favourable for convection -- when the truth is the opposite."""
    cin = np.array([[10.0, np.nan], [np.nan, 250.0]])
    filled, valid = era5.fill_cin(cin)
    assert np.isfinite(filled).all()
    assert (filled[~valid] == era5.CIN_NO_PARCEL_FILL).all()
    assert era5.CIN_NO_PARCEL_FILL >= 1000.0, "must read as strong inhibition"
    assert not (filled[~valid] == 0.0).any()
    assert valid.tolist() == [[True, False], [False, True]]


def test_cin_coverage_is_not_held_to_the_normal_bar():
    """~40% of an India scene has no convective parcel. That is physics, not
    missing data, and must not fail the check."""
    f = era5.ERA5Fields(time="t", lat=np.linspace(5, 40, 8), lon=np.linspace(65, 100, 8),
                        fields={"cin": np.where(np.random.default_rng(0).random((8, 8)) < 0.5,
                                                np.nan, 100.0),
                                "cape": np.full((8, 8), 500.0)})
    rows = dict((n, ok) for n, ok, _ in era5.check_physics(f))
    assert rows["cin coverage"] is True


# --------------------------------------------------------------------------
# ERA5 -- against the live public store
# --------------------------------------------------------------------------

def _has_network():
    """Can we actually READ a chunk, not merely open the metadata?

    The first version called `era5.open_store()` and stopped there. Opening
    the Zarr metadata is a few small object reads and succeeds even when the
    bucket is effectively unreachable; the tests then read a data chunk,
    which blocked forever. The suite hung for 25 minutes and reported
    nothing -- an indefinite block is not even a failure signal.

    Same shape as the test_service_threads gap: the guard exercised a
    DIFFERENT operation from the code it was guarding, so it passed while
    the real path hung. So this reads one scalar, under a deadline.
    """
    if not os.environ.get("NOWCAST_NETWORK_TESTS"):
        return False
    import concurrent.futures as cf

    def probe():
        ds = era5.open_store()
        v = next(iter(era5.REQUIRED_VARS)) if hasattr(era5, "REQUIRED_VARS") \
            else list(ds.data_vars)[0]
        return float(ds[v].isel({d: 0 for d in ds[v].dims}).values)

    try:
        with cf.ThreadPoolExecutor(max_workers=1) as ex:
            ex.submit(probe).result(timeout=30)
        return True
    except Exception:
        return False


# Network tests are OPT-IN: NOWCAST_NETWORK_TESTS=1. A unit suite that
# depends on a live GCS bucket is not a unit suite, and its failure mode
# here was a hang rather than a red test.
network = pytest.mark.skipif(
    not _has_network(),
    reason="ARCO-ERA5 not readable (set NOWCAST_NETWORK_TESTS=1 to enable)")


@network
@pytest.mark.timeout(120)
def test_arco_store_has_every_variable_we_need():
    ds = era5.open_store()
    for v in era5.SINGLE_LEVEL + era5.MULTI_LEVEL:
        assert v in ds, v


@network
@pytest.mark.timeout(120)
def test_real_timestamp_passes_physics():
    f = era5.load_fields("2023-07-09T12:00")
    failures = [(n, d) for n, ok, d in era5.check_physics(f) if not ok]
    assert not failures, failures
    assert set(f.names) == {"cape", "cin", "tcwv", "shear", "convergence",
                            "moisture_convergence"}


@network
@pytest.mark.timeout(120)
def test_cin_undefined_tracks_low_cape_on_real_data():
    """The evidence that CIN NaN means 'no parcel' rather than 'missing'."""
    f = era5.load_fields("2023-07-09T12:00")
    cin, cape = f["cin"], f["cape"]
    nan = ~np.isfinite(cin)
    assert nan.any() and np.isfinite(cin).any()
    assert cape[nan].mean() < cape[~nan].mean() / 5


# --------------------------------------------------------------------------
# IMERG labels
# --------------------------------------------------------------------------

def _two_climates(H=20, W=20, seed=0, n=1200):
    rng = np.random.default_rng(seed)
    scale = np.concatenate([np.full((H, W // 2), 6.0),
                            np.full((H, W // 2), 0.4)], axis=1)
    acc = ClimatologyAccumulator((H, W), n_bins=128)
    for _ in range(n):
        acc.add(rng.exponential(scale), month=7)
    return acc.finalise(train_years=(2016, 2017)), scale, rng


def test_per_cell_thresholds_track_local_climate():
    clim, scale, _ = _two_climates()
    thr = clim.threshold_for(7)
    W = thr.shape[1]
    wet, dry = np.nanmedian(thr[:, :W // 2]), np.nanmedian(thr[:, W // 2:])
    assert wet > 5 * dry, f"wet {wet} vs dry {dry} -- thresholds must be local"


def test_per_cell_thresholds_equalise_the_base_rate():
    """The design goal: without this, a single global threshold would put
    nearly every positive in the wettest region and the model would learn
    geography instead of atmospheric state."""
    clim, scale, rng = _two_climates()
    W = scale.shape[1]
    wet = dry = 0
    for _ in range(300):
        lab, _ = extreme_labels(rng.exponential(scale), clim, 7)
        wet += lab[:, :W // 2].sum()
        dry += lab[:, W // 2:].sum()
    assert 0.5 < (wet / max(dry, 1)) < 2.0, f"wet {wet} vs dry {dry}"


def test_missing_month_raises_rather_than_substituting():
    clim, _, _ = _two_climates()
    with pytest.raises(KeyError, match="do not fall back"):
        clim.threshold_for(3)


def test_cells_without_climatology_are_invalid_not_negative():
    """'We cannot say' is not 'no event' -- scoring them negative would reward
    the model for predicting nothing there."""
    clim = RainClimatology(thresholds={7: np.array([[5.0, np.nan]])})
    lab, valid = extreme_labels(np.array([[9.0, 9.0]]), clim, 7)
    assert lab.tolist() == [[1.0, 0.0]]
    assert valid.tolist() == [[True, False]]


def test_rain_labels_treat_nan_as_invalid_not_dry():
    lab, valid = rain_labels(np.array([[0.5, 2.0, np.nan]]))
    assert lab.tolist() == [[0.0, 1.0, 0.0]]
    assert valid.tolist() == [[True, True, False]]


def test_cloudburst_uses_the_imd_threshold_on_points():
    lab, valid = cloudburst_labels(np.array([99.0, 100.0, 150.0, np.nan]))
    assert lab.tolist() == [0.0, 1.0, 1.0, 0.0]
    assert valid.tolist() == [True, True, True, False]
    assert CLOUDBURST_MM_HR == 100.0


def test_climatology_round_trips(tmp_path):
    clim, _, _ = _two_climates()
    p = tmp_path / "clim.npz"
    clim.save(p)
    back = RainClimatology.load(p)
    assert np.allclose(back.threshold_for(7), clim.threshold_for(7), equal_nan=True)
    assert back.percentile == clim.percentile
    assert tuple(back.train_years) == (2016, 2017)


def test_accumulator_rejects_shape_mismatch():
    acc = ClimatologyAccumulator((4, 4))
    with pytest.raises(ValueError, match="expected"):
        acc.add(np.zeros((5, 5)), month=7)


def test_accumulator_ignores_nan_cells():
    acc = ClimatologyAccumulator((3, 3), n_bins=32)
    a = np.full((3, 3), 2.0); a[0, 0] = np.nan
    for _ in range(50):
        acc.add(a, month=8)
    thr = acc.finalise().threshold_for(8)
    assert np.isnan(thr[0, 0]), "a cell with no valid samples has no threshold"
    assert np.isfinite(thr[1, 1])


# --------------------------------------------------------------------------
# ERA5 verified conventions -- regression guards
# --------------------------------------------------------------------------

def test_negative_longitude_is_rejected_not_silently_empty():
    """ERA5 longitude is 0-360. A negative bound returns an EMPTY selection
    rather than raising, which would surface much later as an unexplained
    all-NaN field."""
    with pytest.raises(ValueError, match="0-360"):
        era5._subset(None, "t", box={"lat": (3, 42), "lon": (-20, 10)})


def test_reversed_longitude_bounds_are_rejected():
    with pytest.raises(ValueError, match="reversed or wrap"):
        era5._subset(None, "t", box={"lat": (3, 42), "lon": (100, 60)})


@network
@pytest.mark.timeout(120)
def test_verified_conventions_still_hold():
    """Pins every convention checked against ERA5 docs AND monsoon physics.
    If ARCO ever changes one of these, this fails instead of the model
    silently learning inverted moisture transport."""
    ds = era5.open_store()
    assert ds.latitude.values[0] > ds.latitude.values[-1], "latitude north-first"
    assert ds.longitude.values.max() > 180, "longitude 0-360"
    assert ds.level.values[0] < ds.level.values[-1], "level top-down"
    assert ds["total_column_water_vapour"].attrs["units"] == "kg m**-2"
    assert ds["convective_inhibition"].attrs["units"] == "J kg**-1"

    # the Somali jet: strong low-level westerlies over the Arabian Sea in July
    sl = ds[["u_component_of_wind"]].sel(
        time="2023-07-09T12:00", level=850,
        latitude=slice(15, 10), longitude=slice(55, 70))
    assert float(np.asarray(sl.u_component_of_wind.values).mean()) > 5.0, \
        "u must be eastward-positive"

    # moisture must be transported TOWARD India, i.e. eastward-positive
    fl = ds["vertical_integral_of_eastward_water_vapour_flux"].sel(
        time="2023-07-09T12:00", latitude=slice(15, 10), longitude=slice(55, 70))
    assert float(np.asarray(fl.values).mean()) > 0, "vapour flux eastward-positive"
