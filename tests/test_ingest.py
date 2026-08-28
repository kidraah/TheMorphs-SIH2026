"""Tests for the Indian ingestion path: credentials, grid, INSAT, IMERG, alignment.

None of these need network access or credentials. They test the decode,
orientation and sanity-check logic against synthetic data shaped like the
real products -- which is where the silent failures live.
"""
import os
import tempfile

import numpy as np
import pytest

from nowcast_data import alignment as al
from nowcast_data import credentials as cr
from nowcast_data import imerg, insat
from nowcast_data.grids import INDIA_4KM, SHAPE, bounds_lonlat, india_area


# --------------------------------------------------------------------------
# Credentials
# --------------------------------------------------------------------------

def test_env_file_is_parsed_without_shell_semantics(tmp_path, monkeypatch):
    p = tmp_path / ".env"
    p.write_text('# comment\nA_KEY=plain\nB_KEY="quoted"\n\nBAD LINE\nC_KEY=$NOT_EXPANDED\n')
    for k in ("A_KEY", "B_KEY", "C_KEY"):
        monkeypatch.delenv(k, raising=False)
    assert cr.load_env(p) == 3
    assert os.environ["A_KEY"] == "plain"
    assert os.environ["B_KEY"] == "quoted"
    assert os.environ["C_KEY"] == "$NOT_EXPANDED", "no shell expansion"


def test_existing_environment_wins_over_env_file(tmp_path, monkeypatch):
    """A CI secret store must not be shadowed by a stale local file."""
    p = tmp_path / ".env"
    p.write_text("X_KEY=from_file\n")
    monkeypatch.setenv("X_KEY", "from_env")
    cr.load_env(p)
    assert os.environ["X_KEY"] == "from_env"
    cr.load_env(p, override=True)
    assert os.environ["X_KEY"] == "from_file"


def test_missing_credential_explains_itself(monkeypatch):
    monkeypatch.setattr(cr, "ENV_FILE", tempfile.mktemp())
    monkeypatch.delenv("MOSDAC_PASSWORD", raising=False)
    with pytest.raises(RuntimeError, match="MOSDAC_PASSWORD"):
        cr.get("MOSDAC_PASSWORD")
    assert cr.get("MOSDAC_PASSWORD", required=False) is None


def test_status_never_returns_secret_values(monkeypatch):
    monkeypatch.setenv("MOSDAC_USERNAME", "supersecret")
    st = cr.status()
    assert st["MOSDAC_USERNAME"] is True
    assert "supersecret" not in cr.report()


# --------------------------------------------------------------------------
# Grid
# --------------------------------------------------------------------------

def test_grid_covers_india_including_the_islands():
    a = india_area()
    for name, lat, lon in [("Kanyakumari", 8.08, 77.55), ("Kashmir", 35.5, 77.0),
                           ("Kibithu", 28.0, 97.4), ("Kutch", 23.7, 68.0),
                           ("Port Blair", 11.6, 92.7)]:
        x, y = a.get_array_indices_from_lonlat(lon, lat)
        assert 0 <= int(x) < a.width and 0 <= int(y) < a.height, name


def test_grid_divides_evenly_for_model_patches_and_labels():
    assert INDIA_4KM.divisible_by(4.0)     # model patch
    assert INDIA_4KM.divisible_by(12.0)    # IMERG-matched labels
    assert not INDIA_4KM.divisible_by(10.0), "10 km would leave fractional blocks"


def test_grid_is_equal_area_not_plate_carree():
    """Area-based scores (CSI, FSS in km) require constant cell area, or a
    '4 km' cell means something different in Kashmir than in Kerala."""
    from nowcast_data.grids import PROJ
    assert PROJ["proj"] == "laea"


# --------------------------------------------------------------------------
# INSAT sanity checks
# --------------------------------------------------------------------------

def _plausible_scan(seed=0):
    rng = np.random.default_rng(seed)
    return {"TIR1": rng.uniform(200, 300, (60, 60)),
            "WV": rng.uniform(215, 260, (60, 60))}


def test_plausible_scan_passes():
    assert insat.check_physics(_plausible_scan()).passed


def test_celsius_instead_of_kelvin_is_caught():
    """The classic calibration slip: values look like weather, units do not."""
    s = {k: v - 273.15 for k, v in _plausible_scan().items()}
    chk = insat.check_physics(s)
    assert not chk.passed
    assert "wrong units" in chk.report()


def test_reflectance_mistaken_for_temperature_is_caught():
    s = {"TIR1": np.random.default_rng(0).uniform(0, 1, (40, 40))}
    assert not insat.check_physics(s).passed


def test_water_vapour_must_be_colder_and_narrower_than_tir():
    """The window/absorption contrast. WV sees only the upper troposphere --
    the same signature confirmed in SEVIR's ir069."""
    rng = np.random.default_rng(1)
    swapped = {"TIR1": rng.uniform(215, 260, (50, 50)),
               "WV": rng.uniform(200, 300, (50, 50))}
    chk = insat.check_physics(swapped)
    assert not chk.passed
    assert any("narrower" in r.name and not r.passed for r in chk.results)


def test_empty_scan_is_caught():
    chk = insat.check_physics({"TIR1": np.full((20, 20), np.nan)})
    assert not chk.passed


def test_strict_mode_raises_on_bad_physics():
    with pytest.raises(ValueError, match="failed sanity checks"):
        insat.check_physics({"TIR1": np.zeros((10, 10))}).raise_if_failed()


def test_expected_resolutions_match_the_satpy_reader():
    """satpy's insat3d_img_l1b_h5 declares 1000/4000/8000 m; the whole
    per-channel treatment rests on TIR being 4 km and WV 8 km."""
    assert insat.INSAT_RESOLUTION_M["TIR1"] == 4000
    assert insat.INSAT_RESOLUTION_M["WV"] == 8000


def test_fetch_is_not_silently_faked():
    with pytest.raises(NotImplementedError, match="mosdac"):
        insat.fetch_scan()


# --------------------------------------------------------------------------
# IMERG
# --------------------------------------------------------------------------

def _write_imerg(tmp_path, var="Grid/precipitation", value=12.5,
                 lon_slice=slice(2600, 2620), lat_slice=slice(1150, 1170),
                 dry_rest=True):
    import h5py
    p = tmp_path / "imerg.HDF5"
    fill = 0.0 if dry_rest else imerg.FILL_VALUE
    a = np.full((1,) + imerg.IMERG_SHAPE_LONLAT, fill, dtype=np.float32)
    a[0, lon_slice, lat_slice] = value
    with h5py.File(p, "w") as f:
        f.create_dataset(var, data=a)
    return p


def test_lon_first_storage_is_transposed_to_lat_lon(tmp_path):
    """IMERG stores (time, lon, lat) -- longitude FIRST, unlike almost
    everything else. Missing the transpose puts rain in the wrong place
    while still looking like weather."""
    s = imerg.read_precipitation(_write_imerg(tmp_path))
    assert s.rate_mmhr.shape == imerg.IMERG_SHAPE_LONLAT[::-1]

    ys, xs = np.where(s.rate_mmhr > 1.0)
    assert 24 < s.lats[ys.min()] < 28, "rain block must land at the right latitude"
    assert 79 < s.lons[xs.min()] < 83, "and the right longitude"


def test_fill_value_becomes_nan(tmp_path):
    p = _write_imerg(tmp_path, dry_rest=False)
    s = imerg.read_precipitation(p)
    assert np.isnan(s.rate_mmhr).any()
    assert np.nanmin(s.rate_mmhr) >= 0.0, "-9999.9 must not survive as rain"


def test_v06_variable_name_still_works(tmp_path):
    s = imerg.read_precipitation(_write_imerg(tmp_path, var="Grid/precipitationCal"))
    assert s.variable == "Grid/precipitationCal"


def test_unknown_variable_name_reports_what_is_present(tmp_path):
    import h5py
    p = tmp_path / "bad.HDF5"
    with h5py.File(p, "w") as f:
        f.create_dataset("Grid/somethingElse", data=np.zeros((1, 4, 4)))
    with pytest.raises(KeyError, match="precipitation"):
        imerg.read_precipitation(p)


def test_wrong_grid_shape_is_rejected_rather_than_transposed(tmp_path):
    import h5py
    p = tmp_path / "sub.HDF5"
    with h5py.File(p, "w") as f:
        f.create_dataset("Grid/precipitation", data=np.zeros((1, 100, 50)))
    with pytest.raises(ValueError, match="0.1 deg global"):
        imerg.read_precipitation(p)


def test_coverage_check_requires_a_real_fraction(tmp_path):
    """'some finite pixel exists' is too weak -- a real scan is near-fully
    populated because dry is 0.0, not missing."""
    s = imerg.read_precipitation(_write_imerg(tmp_path, dry_rest=False))
    checks = dict((n, ok) for n, ok, _ in imerg.check_physics(s))
    assert checks["coverage"] is False


def test_negative_rain_is_caught(tmp_path):
    s = imerg.read_precipitation(_write_imerg(tmp_path))
    s.rate_mmhr[0, 0] = -5.0
    checks = dict((n, ok) for n, ok, _ in imerg.check_physics(s))
    assert checks["no negative rain"] is False


def test_imerg_fetch_is_not_silently_faked():
    with pytest.raises(NotImplementedError, match="GES DISC"):
        imerg.fetch()


# --------------------------------------------------------------------------
# Alignment -- the gate before Indian training data
# --------------------------------------------------------------------------

def _cells(shift_px=0, seed=0, n=12, size=200):
    rng = np.random.default_rng(seed)
    tir = np.full((size, size), 285.0)
    rain = np.zeros((size, size))
    yy, xx = np.mgrid[0:size, 0:size]
    for _ in range(n):
        y0, x0 = rng.integers(25, size - 25), rng.integers(25, size - 25)
        d2 = (yy - y0) ** 2 + (xx - x0) ** 2
        tir[d2 < 80] = 215.0
        rain[d2 < 80] = 4.0
    return tir, np.roll(rain, shift_px, axis=1)


def test_aligned_fields_peak_at_zero_offset():
    r = al.alignment_offset(*_cells(0))
    assert (r.dy, r.dx) == (0, 0)
    assert r.aligned and "ALIGNED" in r.report()


def test_systematic_shift_is_detected_with_the_right_magnitude():
    """The failure mode: the model would learn 'rain appears 28 km east of
    the cold top' -- stable, learnable, and completely wrong."""
    r = al.alignment_offset(*_cells(7))
    assert r.dx == -7 and r.dy == 0
    assert r.offset_km == (0.0, -28.0)
    assert not r.aligned
    assert "MISALIGNED" in r.report() and "Do NOT build training data" in r.report()


def test_one_pixel_slack_is_tolerated():
    """Convective rain genuinely lags the cloud top slightly; demanding an
    exact zero would reject correctly-georeferenced data."""
    assert al.alignment_offset(*_cells(1)).aligned


def test_mismatched_grids_are_rejected():
    tir, rain = _cells(0)
    with pytest.raises(ValueError, match="same grid"):
        al.alignment_offset(tir, rain[:100])


def test_scan_without_convection_says_so():
    tir = np.full((50, 50), 290.0)
    with pytest.raises(ValueError, match="no pixels colder"):
        al.alignment_offset(tir, np.ones((50, 50)) * 5)


def test_scan_without_rain_says_so():
    tir = np.full((50, 50), 210.0)
    with pytest.raises(ValueError, match="no pixels wetter"):
        al.alignment_offset(tir, np.zeros((50, 50)))
