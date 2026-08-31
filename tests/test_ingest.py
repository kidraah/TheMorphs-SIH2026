"""Tests for the Indian ingestion path: credentials, grid, INSAT, IMERG, alignment.

None need network access or credentials. Most run against synthetic data
shaped like the real products -- which is where the silent failures live --
and the final block runs against the real 2019 MOSDAC scan when it is on
disk, skipping otherwise.
"""
import os
from pathlib import Path
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

def _realistic_tir(shape=(80, 80), seed=0):
    """A field whose PERCENTILES are plausible, not just its mean.

    A constant 280 K array is not a valid TIR scan: its p1 and p99.9 are both
    280, which the percentile checks correctly reject. Fixtures have to span
    a realistic distribution now.
    """
    rng = np.random.default_rng(seed)
    a = rng.normal(281.0, 11.0, shape)          # surface + low/mid cloud
    n = a.size // 9
    idx = rng.choice(a.size, n, replace=False)
    a.flat[idx] = rng.normal(208.0, 12.0, n)    # cold convective tops
    return a
    # deliberately NOT np.clip'd: clipping piles pixels at exactly the bound,
    # which is a clamp, and the saturation check correctly flags it. That is
    # how this fixture was caught the first time.


def _realistic_wv(shape=(80, 80), seed=1):
    rng = np.random.default_rng(seed)
    return rng.normal(241.0, 11.0, shape)


def _plausible_scan(seed=0):
    return {"TIR1": _realistic_tir(seed=seed), "WV": _realistic_wv(seed=seed + 1)}


def test_plausible_scan_passes():
    chk = insat.check_physics(_plausible_scan())
    assert chk.passed, chk.report()


def test_the_fixture_is_actually_realistic():
    """Guard: if the synthetic scan stops resembling real percentiles, the
    checks above stop testing anything meaningful."""
    a = _realistic_tir()
    assert 185 <= np.percentile(a, 1) <= 245
    assert 285 <= np.percentile(a, 99.9) <= 315


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


def test_imerg_granule_url_encodes_the_window_and_index():
    """IMERG filenames carry a half-open window AND a minutes-since-midnight
    index. Getting the index wrong yields a 404, not a wrong file."""
    u = imerg.granule_url("2025-08-01T23:45")
    assert "/2025/213/" in u, "day-of-year 213 for 1 Aug 2025"
    assert "S233000-E235959" in u, "the granule containing 23:45"
    assert ".1410." in u, "1410 minutes since midnight = 23:30"
    assert u.endswith(".HDF5")


def test_granule_url_floors_to_the_containing_half_hour():
    for t, want in (("2025-08-01T00:00", "S000000-E002959"),
                    ("2025-08-01T00:29", "S000000-E002959"),
                    ("2025-08-01T00:30", "S003000-E005959"),
                    ("2025-08-01T12:15", "S120000-E122959")):
        assert want in imerg.granule_url(t), t


REAL_IMERG = Path("data/imerg/3B-HHR.MS.MRG.3IMERG.20250801-S233000-E235959"
                  ".1410.V07B.HDF5")


@pytest.mark.skipif(not REAL_IMERG.exists(), reason="IMERG granule not downloaded")
def test_real_imerg_granule_passes_every_check():
    """Sixth data source through the standing rule. Clean."""
    s = imerg.read_precipitation(REAL_IMERG)
    failures = [(n, d) for n, ok, d in imerg.check_physics(s) if not ok]
    assert not failures, failures
    assert s.variable == "Grid/precipitation"


@pytest.mark.skipif(not REAL_IMERG.exists(), reason="IMERG granule not downloaded")
def test_real_imerg_zero_mode_is_not_flagged_as_a_sentinel():
    """87% of a real rain field is exactly 0.0. That is the legitimate dry
    boundary mode, and the detector must not call it a fill value."""
    s = imerg.read_precipitation(REAL_IMERG)
    r = detect_pileups(s.rate_mmhr[np.isfinite(s.rate_mmhr)], min_fraction=0.0005)
    assert not r.has_suspicious, r.report("imerg")
    assert r.pileups[0].value == 0.0 and r.pileups[0].fraction > 0.5


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


# --------------------------------------------------------------------------
# Night gating and product-version reporting
# --------------------------------------------------------------------------

def test_reflective_channels_are_na_at_night_not_failed():
    """23:30 UTC is 05:00 IST. A dark VIS channel is correct behaviour, and
    scoring it FAIL would make every pre-dawn scan look broken."""
    meta = {"Sun_Elevation(Degrees)": -12.56}
    arrays = {"TIR1": _realistic_tir(),
              "VIS": np.zeros((30, 30)), "SWIR": np.zeros((30, 30))}
    chk = insat.check_physics(arrays, metadata=meta)
    assert chk.passed, "night-time VIS must not fail the scan"
    reflective_skips = {r.name.split()[0] for r in chk.results
                        if r.skipped and "(reflective)" in r.name}
    assert reflective_skips == {"VIS", "SWIR"}
    assert "N/A" in chk.report()


def test_reflective_channels_are_checked_in_daylight():
    meta = {"Sun_Elevation(Degrees)": 62.0}
    arrays = {"TIR1": _realistic_tir(), "VIS": np.zeros((30, 30))}
    chk = insat.check_physics(arrays, metadata=meta)
    reflective_skips = [r for r in chk.results
                        if r.skipped and "(reflective)" in r.name]
    assert not reflective_skips, "reflective checks must run when the sun is up"


def test_illumination_is_reported_either_way():
    chk = insat.check_physics({"TIR1": _realistic_tir()},
                              metadata={"Sun_Elevation(Degrees)": -12.56})
    assert any("sun elevation" in r.detail for r in chk.results)
    assert insat.is_daylight({"Sun_Elevation(Degrees)": -12.56}) is False
    assert insat.is_daylight({"Sun_Elevation(Degrees)": 45.0}) is True
    assert insat.is_daylight({}) is True, "absent metadata must not skip silently"


def test_skipped_checks_do_not_mask_a_real_failure():
    meta = {"Sun_Elevation(Degrees)": -12.0}
    chk = insat.check_physics({"TIR1": np.full((30, 30), 5.0),      # bad
                               "VIS": np.zeros((30, 30))}, metadata=meta)
    assert not chk.passed


def test_product_identity_is_reported_with_the_result():
    """A PASS must be attributable to a SPECIFIC product version: a 2019
    file passing says nothing about MOSDAC's current output."""
    meta = {"Software_Version": "1.0", "Product_Type": "STANDARD(FULL DISK)",
            "HDF_Product_File_Name": "3DIMG_25AUG2019_2330_L1B_STD.h5"}
    txt = insat.check_physics({"TIR1": _realistic_tir()},
                              metadata=meta).report()
    assert "Software_Version" in txt and "1.0" in txt
    assert "diff this against a current-version file" in txt


def test_checks_are_on_percentiles_not_extremes():
    """Regression on a real mistake made on this project.

    The absolute min/max of a valid scan are contaminated: TIR1's 330.6 K
    maximum is scattered singleton noise 28 K above p99.99, and its 180.09 K
    floor is the count->temperature LUT CLAMPING (LUT[-1] == LUT[-2]), which
    is why WV independently 'bottoms out' at almost the same value despite
    seeing only the upper troposphere.

    Widening the bounds to admit those artefacts -- which is what was done
    first -- fits the check to the data and defeats its purpose. The measured
    robust percentiles must sit comfortably inside the bounds instead.
    """
    tir = insat.EXPECTED_BT_PERCENTILES["TIR1"]
    lo, hi = tir[1.0]
    assert lo <= 195.4 <= hi, "measured TIR1 p1"
    lo, hi = tir[99.9]
    assert lo <= 301.1 <= hi, "measured TIR1 p99.9"

    wv = insat.EXPECTED_BT_PERCENTILES["WV"]
    lo, hi = wv[1.0]
    assert lo <= 209.7 <= hi, "measured WV p1"
    lo, hi = wv[99.9]
    assert lo <= 266.4 <= hi, "measured WV p99.9"

    # and the contaminated extremes must NOT be admitted as valid values
    assert not (tir[1.0][0] <= 180.09), "the LUT clamp must not be inside the bound"
    assert not (tir[99.9][1] >= 330.56), "the noise spike must not be inside the bound"


def test_lut_clamp_contamination_is_reported():
    """A large spike of pixels at exactly one value is a clamp or an unmasked
    fill, not signal -- the same class as the VIL byte-255 and int16 sentinel
    bugs already found on this project."""
    a = _realistic_tir((100, 100))
    a[:40] = a.min()                     # 40% clamped at the floor
    chk = insat.check_physics({"TIR1": a}, metadata={})
    sat = [r for r in chk.results if "floor saturation" in r.name]
    assert sat and not sat[0].passed
    assert "LUT clamp" in sat[0].detail


def test_extremes_are_reported_as_context_not_pass_fail():
    a = _realistic_tir((50, 50))
    a[0, 0] = 400.0                      # a single absurd pixel
    chk = insat.check_physics({"TIR1": a}, metadata={})
    ext = [r for r in chk.results if "extremes" in r.name]
    assert ext and ext[0].skipped, "extremes must be context, never a verdict"


# --------------------------------------------------------------------------
# Against the real MOSDAC file, when present
# --------------------------------------------------------------------------

REAL_INSAT = Path("/Users/evad/chuchuchuchu/3DIMG_L1B_STD/2019/25AUG/"
                  "3DIMG_25AUG2019_2330_L1B_STD_V01R00.h5")


@pytest.mark.skipif(not REAL_INSAT.exists(), reason="real INSAT file not present")
def test_real_2019_scan_passes_end_to_end():
    meta = insat.read_metadata(REAL_INSAT)
    assert meta["Satellite_Name"] == "INSAT-3D"
    assert meta["Processing_Level"] == "L1B"
    assert insat.solar_elevation(meta) < 0, "23:30 UTC is pre-dawn over India"

    scn = insat.read_scan(REAL_INSAT, ("TIR1", "WV"))
    native = {c: np.asarray(scn[c].values, dtype=np.float64) for c in ("TIR1", "WV")}
    chk = insat.check_physics(native, metadata=meta)
    assert chk.passed, chk.report()

    # the resolution hierarchy the whole pipeline rests on
    assert native["TIR1"].shape == (2816, 2805)
    assert native["WV"].shape == (1408, 1402)
    assert native["TIR1"].shape[0] == 2 * native["WV"].shape[0]


# --------------------------------------------------------------------------
# Standing sentinel audit
# --------------------------------------------------------------------------

from nowcast_data.sentinels import detect_pileups


def test_catches_all_three_bugs_already_found_by_hand():
    """Regression on this project's most repeated failure. All three decoded
    to plausible physical EXTREMES and none of them raised."""
    rng = np.random.default_rng(0)

    # 1. SEVIR VIL byte 255 -> 81.33 kg/m^2, top of the range
    raw = rng.integers(0, 60, 100_000).astype(np.uint8); raw[:27_000] = 255
    vil = np.where(raw > 18, np.exp((raw.astype(float) - 83.9) / 38.9),
                   np.where(raw > 5, (raw.astype(float) - 2) / 90.66, 0.0))
    assert detect_pileups(vil).has_suspicious

    # 2. int16 missing sentinel -> -327.68 degC, coldest possible cloud top
    tir = rng.normal(280, 15, 100_000); tir[:7_000] = -327.68
    assert detect_pileups(tir).has_suspicious

    # 3. INSAT count->K LUT clamp at only 0.33% of pixels
    k = rng.normal(281, 11, 300_000); k[:1000] = 180.09
    assert detect_pileups(k).has_suspicious


def test_legitimate_dry_pixel_zero_mode_is_not_flagged():
    """The control that must not fire: an IMERG rain field is mostly zeros,
    and zero is both the natural boundary AND the modal value. An earlier
    version flagged it HIGH because the IQR of a mostly-zero field is
    degenerate, inflating the isolation score to 2e8."""
    rng = np.random.default_rng(0)
    rain = np.where(rng.random(200_000) < 0.85, 0.0, rng.exponential(3, 200_000))
    r = detect_pileups(rain)
    assert not r.has_suspicious, r.report("rain")
    assert r.pileups and r.pileups[0].value == 0.0
    assert "natural boundary" in r.pileups[0].note


def test_clean_field_is_clean():
    rng = np.random.default_rng(1)
    assert not detect_pileups(rng.normal(280, 15, 100_000)).has_suspicious


def test_isolation_not_mass_is_the_signal():
    """A huge but continuous mode is fine; a small isolated spike is not."""
    rng = np.random.default_rng(2)
    continuous = np.concatenate([np.full(90_000, 0.0), rng.exponential(2, 10_000)])
    spike = rng.normal(280, 15, 100_000).copy(); spike[:500] = -9999.9
    assert not detect_pileups(continuous).has_suspicious
    assert detect_pileups(spike).has_suspicious, "0.5% isolated spike must flag"


def test_pileup_scan_runs_inside_the_insat_checker():
    a = _realistic_tir((200, 200))
    a[:20] = -9999.9                      # unmasked fill
    chk = insat.check_physics({"TIR1": a}, metadata={})
    row = [r for r in chk.results if "pile-up" in r.name]
    assert row and not row[0].passed
    assert "check the product docs" in row[0].detail


def test_pileup_scan_runs_inside_the_imerg_checker(tmp_path):
    s = imerg.read_precipitation(_write_imerg(tmp_path))
    s.rate_mmhr[:200] = 12345.0           # an isolated spike
    names = [n for n, ok, _ in imerg.check_physics(s) if n == "value pile-up"]
    assert names, "IMERG checker must run the sentinel scan"


# --------------------------------------------------------------------------
# Cross-satellite consistency
# --------------------------------------------------------------------------

from nowcast_data.cross_satellite import compare_distributions


def _bt(n=1_000_000, seed=0):
    """~1M pixels. A real INSAT TIR scan is 2816x2805 = 7.9M, and below
    ~500k the p99.9 comparison is sampling noise (1.31 K at n=100k)."""
    return np.random.default_rng(seed).normal(275.0, 20.0, n)


def test_identical_instruments_read_as_consistent():
    a, b = _bt(seed=0), _bt(seed=1)
    r = compare_distributions({"TIR1": a}, {"TIR1": b})
    assert r.safe_to_mix
    assert not r.needs_single_satellite
    assert "train across satellites" in r.report()


def test_constant_offset_is_diagnosed_as_correctable():
    """A calibration offset is fixable by per-satellite normalisation."""
    a = _bt()
    r = compare_distributions({"TIR1": a}, {"TIR1": a + 1.2})
    c = r.comparisons[0]
    assert c.verdict == "offset"
    assert c.is_offset_like
    assert abs(c.median_diff - 1.2) < 0.05
    assert not r.safe_to_mix and not r.needs_single_satellite
    assert "normalisation" in r.report()


def test_calibration_slope_forces_a_single_satellite():
    """A slope is NOT correctable by shifting the mean -- the distributions
    differ in shape, so the model could learn satellite identity."""
    a = _bt()
    r = compare_distributions({"TIR1": a}, {"TIR1": 275 + (a - 275) * 1.15})
    c = r.comparisons[0]
    assert not c.is_offset_like, "a slope must not be mistaken for an offset"
    assert c.verdict in ("shape-differs", "large")
    assert r.needs_single_satellite
    assert "3RIMG" in r.report(), "should name the satellite to fall back to"


def test_grids_of_different_shapes_are_comparable():
    """3DR and 3DS view from different sub-satellite longitudes, so a
    pixelwise difference would confound geometry with calibration.
    Distributions must be comparable regardless of grid shape."""
    r = compare_distributions({"TIR1": _bt(1_000_000, 0).reshape(2000, 500)},
                              {"TIR1": _bt(800_000, 1).reshape(1600, 500)})
    assert r.comparisons and r.comparisons[0].verdict == "consistent"


def test_small_samples_are_reported_inconclusive_not_different():
    """Guard against a spurious verdict: two IDENTICAL instruments sampled at
    100k pixels differ at p99.9 by ~1.3 K from noise alone, which would
    otherwise read as 'shape-differs' and wrongly rule out mixing."""
    r = compare_distributions({"TIR1": _bt(100_000, 0)}, {"TIR1": _bt(100_000, 1)})
    c = r.comparisons[0]
    assert c.underpowered
    assert c.verdict == "underpowered"
    assert not r.needs_single_satellite, "noise must not masquerade as a slope"
    assert "INCONCLUSIVE" in r.report()


def test_channels_present_in_only_one_scene_are_skipped():
    r = compare_distributions({"TIR1": _bt(), "WV": _bt(seed=2)},
                              {"TIR1": _bt(seed=1)})
    assert [c.channel for c in r.comparisons] == ["TIR1"]


def test_too_few_pixels_is_not_compared():
    r = compare_distributions({"TIR1": np.array([275.0, 276.0])},
                              {"TIR1": np.array([275.0, 276.0])})
    assert r.comparisons == []


# --------------------------------------------------------------------------
# Findings from the 2025 3DR/3DS gate files
# --------------------------------------------------------------------------

def test_denormal_solar_elevation_is_rejected():
    """INSAT-3DS writes Sun_Elevation(Degrees) = 7.68e-76. It is denormal-small
    rather than out of range, so a naive float() accepts it and reads as
    '0 degrees' -- which gates a night scan correctly by luck and would let
    reflective checks run on a daytime scene with unknown illumination."""
    assert insat.solar_elevation({"Sun_Elevation(Degrees)": 7.68248614658878e-76}) is None
    assert insat.solar_elevation({"Sun_Elevation(Degrees)": -17.26}) == pytest.approx(-17.26)
    assert insat.solar_elevation({"Sun_Elevation(Degrees)": 1e3}) is None
    assert insat.solar_elevation({"Sun_Elevation(Degrees)": float("nan")}) is None


def test_small_scan_sample_size_is_flagged_not_failed():
    """p1/p99.9 on a crop carry ~1 K of sampling noise. Flag it, don't fail:
    the same effect measured for cross-satellite comparison applies here."""
    chk = insat.check_physics({"TIR1": _realistic_tir((100, 100))}, metadata={})
    rows = [r for r in chk.results if "sample size" in r.name]
    assert rows and rows[0].skipped
    assert "sampling noise" in rows[0].detail
    assert chk.passed, "a small sample is a caveat, not a failure"


REAL_3DR = Path("/Users/evad/test_ff/3RIMG_L1B_STD/2025/01AUG/"
                "3RIMG_01AUG2025_2345_L1B_STD_V01R00.h5")
REAL_3DS = Path("/Users/evad/test_ff/3SIMG_L1B_STD/2025/01AUG/"
                "3SIMG_01AUG2025_2330_L1B_STD_V01R00.h5")


@pytest.mark.skipif(not REAL_3DR.exists(), reason="3DR gate file not present")
def test_3dr_passes_through_satpy():
    """3DR is the workhorse: every hindcast event from 2016 and all 24 archive
    configs route through it."""
    arrays, chk = insat.ingest_scan(REAL_3DR, ("TIR1", "WV"), strict=False)
    assert chk.passed, chk.report()
    assert chk.metadata["_reader"] == "satpy"
    assert all(v.shape == (912, 864) for v in arrays.values())


@pytest.mark.skipif(not REAL_3DS.exists(), reason="3DS gate file not present")
def test_3ds_needs_the_native_reader():
    """satpy 0.60 raises KeyError('Longitude_WV') on 3DS: its WV channel was
    upgraded to full 4 km resolution, so the separate coarse WV geolocation
    arrays that 3D/3DR carry no longer exist."""
    with pytest.raises(KeyError, match="Longitude_WV|available"):
        insat.read_scan(REAL_3DS, ("TIR1", "WV"))

    arrays, chk = insat.ingest_scan(REAL_3DS, ("TIR1", "WV"), strict=False)
    assert chk.passed, chk.report()
    assert chk.metadata["_reader"].startswith("native")
    assert all(v.shape == (912, 864) for v in arrays.values())


@pytest.mark.skipif(not REAL_3DS.exists(), reason="3DS gate file not present")
def test_3ds_water_vapour_is_full_resolution():
    """The structural change behind the satpy failure, and a live design
    assumption: 'INSAT WV is 8 km' is true for 3D/3DR and FALSE for 3DS."""
    d = insat.read_scan_native(REAL_3DS, ("TIR1", "WV"))
    assert d["WV"]["bt"].shape == d["TIR1"]["bt"].shape == (2816, 2805)
    assert d["WV"]["geo_source"] == "Latitude", "3DS WV shares the main geolocation"


@pytest.mark.skipif(not REAL_3DR.exists(), reason="3DR gate file not present")
def test_3dr_water_vapour_is_half_resolution():
    d = insat.read_scan_native(REAL_3DR, ("TIR1", "WV"))
    assert d["TIR1"]["bt"].shape == (2816, 2805)
    assert d["WV"]["bt"].shape == (1408, 1402)
    assert d["WV"]["geo_source"] == "Latitude_WV"


@pytest.mark.skipif(not REAL_3DR.exists(), reason="3DR gate file not present")
def test_disk_median_solar_elevation_would_be_wrong():
    """A geostationary disk always spans the terminator, so its median
    elevation is ~0 by construction -- +0.80 deg for a scan whose attribute
    reads -17.26. Sampling must be over the AOI, not the disk."""
    aoi = insat.solar_elevation_from_dataset(REAL_3DR, aoi_lat=23.0, aoi_lon=82.0)
    assert aoi is not None and aoi < 0, f"India pre-dawn should be negative, got {aoi}"


# --------------------------------------------------------------------------
# Alignment gate
# --------------------------------------------------------------------------

from nowcast_data import alignment_gate as ag


def _synth_scenes(const_dy=0, const_dx=0, parallax=True, n=8, cold=5000, wet=4000,
                  peak=0.42, zero=0.28, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        lat = 10 + 3 * i
        tz = float(np.tan(np.deg2rad(ag.satellite_zenith(lat, 78.0))))
        dy = const_dy + (12.0 * tz / 4.0 if parallax else 0.0)
        out.append(ag.SceneOffset(f"s{i}", int(round(dy)), int(round(const_dx)),
                                  peak=peak, zero=zero, n_cold=cold, n_wet=wet,
                                  mean_tan_zenith=tz))
    return out


def test_parallax_alone_does_not_fail_the_gate():
    """A naive one-pixel gate would FAIL correct data: a 12 km cloud top is
    displaced 2.5 px at Kashmir by viewing geometry alone."""
    r = ag.run_gate(_synth_scenes())
    assert r.verdict == "PASS", r.report()
    assert abs(r.parallax_slope_km - 12.0) < 5.0, "parallax slope should be recovered"


def test_constant_offset_fails_and_is_named_as_georeferencing():
    r = ag.run_gate(_synth_scenes(const_dy=5))
    assert r.verdict == "FAIL"
    assert any("georeferencing error" in x for x in r.reasons)
    assert "DO NOT BUILD TRAINING DATA" in r.report()


def test_parallax_magnitude_matches_geometry():
    """Displacement must grow with satellite zenith angle, i.e. with distance
    from the sub-satellite point."""
    south = ag.expected_parallax_km(8.1, 77.5)
    north = ag.expected_parallax_km(34.2, 75.5)
    assert north > 3 * south
    assert 8.0 < north < 13.0, f"Kashmir 12 km top should shift ~10 km, got {north}"


def test_too_few_scenes_is_inconclusive_not_pass():
    r = ag.run_gate(_synth_scenes(n=2))
    assert r.verdict == "INCONCLUSIVE"
    assert "cannot separate a registration offset from storm motion" in r.reasons[0]


def test_flat_peak_is_inconclusive_not_pass():
    """No convection means no alignment signal. That must not read as PASS."""
    r = ag.run_gate(_synth_scenes(peak=0.28, zero=0.28))
    assert r.verdict == "INCONCLUSIVE"
    assert "no convection to align" in r.reasons[0]


def test_thin_evidence_scenes_are_excluded():
    r = ag.run_gate(_synth_scenes(cold=10, wet=10))
    assert r.verdict == "INCONCLUSIVE"


def test_inconsistent_offsets_fail_even_if_the_median_is_small():
    """A registration error is consistent. Large scatter means storm motion is
    being measured, and a small median would otherwise hide it."""
    s = _synth_scenes(parallax=False)
    for i, sc in enumerate(s):
        sc.dy = 8 if i % 2 else -8
        sc.mean_tan_zenith = 0.3
    r = ag.run_gate(s)
    assert r.verdict == "FAIL"
    assert any("scatter" in x or "spread" in x for x in r.reasons)


def test_cellularity_prefers_many_cores_over_one_broad_shield():
    """What identifies an alignment offset is CELLULAR convection. A broad
    stratiform shield overlaps at any small shift and gives a flat peak --
    measured at 1.04x on the real 2345Z scene."""
    broad = np.zeros((200, 200)); broad[60:140, 60:140] = 8.0      # one shield
    cellular = np.zeros((200, 200))
    rng = np.random.default_rng(0)
    for _ in range(40):                                            # many cores
        y, x = rng.integers(10, 190, 2)
        cellular[y-3:y+3, x-3:x+3] = 8.0

    b = ag.scene_cellularity(broad)
    c = ag.scene_cellularity(cellular)
    assert c["cores"] > 10 * b["cores"]
    assert c["score"] > b["score"], "cellular must outrank broad rain"


def test_dry_scene_scores_zero():
    assert ag.scene_cellularity(np.zeros((50, 50)))["score"] == 0.0


def test_ranking_picks_the_most_cellular():
    rng = np.random.default_rng(1)
    scored = {}
    for i, ncore in enumerate((2, 40, 15)):
        a = np.zeros((150, 150))
        for _ in range(ncore):
            y, x = rng.integers(5, 145, 2)
            a[y-2:y+2, x-2:x+2] = 9.0
        scored[f"t{i}"] = ag.scene_cellularity(a)
    assert ag.rank_candidate_scenes(scored, 1)[0][0] == "t1"


# --------------------------------------------------------------------------
# Static zenith channels
# --------------------------------------------------------------------------

from nowcast_data.static_channels import expected_shift_px, zenith_channels


def test_zenith_channel_supplies_direction_not_just_magnitude():
    """Parallax has a bearing: away from the sub-satellite point. West of it
    the shift is WESTWARD. Magnitude alone would lose that, leaving the model
    to infer the bearing from position -- the inference we are saving it."""
    from nowcast_data.grids import india_area
    ch = zenith_channels(sub_lon=74.0)
    area = india_area()

    def at(lat, lon, k):
        x, y = area.get_array_indices_from_lonlat(lon, lat)
        return float(ch[k][int(y), int(x)])

    assert at(23.7, 68.5, "parallax_dx") < 0, "west of 74E shifts westward"
    assert at(22.6, 88.4, "parallax_dx") > 0, "east of 74E shifts eastward"
    assert at(34.2, 75.5, "parallax_dy") > at(8.1, 77.5, "parallax_dy"), \
        "northward shift grows with latitude"


def test_zenith_channel_matches_measured_geometry():
    from nowcast_data.grids import india_area
    ch = zenith_channels(sub_lon=74.0)
    area = india_area()
    x, y = area.get_array_indices_from_lonlat(75.5, 34.2)
    shift_px = float(ch["tan_zenith"][int(y), int(x)]) * 12.0 / 4.0
    assert 2.2 < shift_px < 2.8, f"Kashmir 12 km top should shift ~2.5 px, got {shift_px}"


def test_channels_are_on_the_analysis_grid():
    ch = zenith_channels()
    assert all(v.shape == (912, 864) for v in ch.values())
    assert np.isfinite(ch["tan_zenith"]).all()


# ---------------------------------------------------------------------------
# Tiled alignment measurement (the gate's lever arm)
# ---------------------------------------------------------------------------

def _cellular_scene(shift=5, n_cores=300, seed=3):
    """A scene that looks like identifiable convection: many separate cores,
    with the rain rigidly displaced by a KNOWN amount."""
    import numpy as np
    from nowcast_data.grids import india_area
    a = india_area(); lons, lats = a.get_lonlats()
    rng = np.random.default_rng(seed)
    h, w = lats.shape
    tir = np.full((h, w), 280.0); rain = np.zeros((h, w))
    for _ in range(n_cores):
        r, c = rng.integers(150, h - 150), rng.integers(150, w - 150)
        rr = int(rng.integers(6, 14))
        tir[r - rr:r + rr, c - rr:c + rr] = 215.0
        rain[r - rr + shift:r + rr + shift, c - rr + shift:c + rr + shift] = 8.0
    return tir, rain, lats, lons


def test_tiles_recover_a_known_displacement_on_a_cellular_scene():
    from nowcast_data.alignment_gate import measure_scene_tiled
    tir, rain, lats, lons = _cellular_scene(shift=5)
    tiles = measure_scene_tiled(tir, rain, lats, lons, scene="t")
    assert len(tiles) >= 10
    assert all((t.dy, t.dx) == (-5, -5) for t in tiles), \
        sorted({(t.dy, t.dx) for t in tiles})
    assert all(t.identifiable for t in tiles)


def test_tiling_widens_the_zenith_lever_arm_by_an_order_of_magnitude():
    """The whole reason tiles exist: between-scene tan(zenith) sd is 0.041,
    which cannot resolve a 12 km parallax slope at ANY scene count."""
    from nowcast_data.alignment_gate import measure_scene_tiled
    tir, rain, lats, lons = _cellular_scene()
    tz = [t.mean_tan_zenith for t in measure_scene_tiled(tir, rain, lats, lons)]
    assert max(tz) - min(tz) > 0.4, max(tz) - min(tz)


def test_the_rain_field_is_not_masked_to_the_tile():
    """Masking both fields means any shift moves the rain off the tile, so
    zero offset always wins and the gate can never fail. Measured: every
    interior band returned (0, 0) for a known (-5, -5) displacement."""
    from nowcast_data.alignment_gate import measure_scene_tiled
    tir, rain, lats, lons = _cellular_scene(shift=5)
    tiles = measure_scene_tiled(tir, rain, lats, lons)
    assert not any((t.dy, t.dx) == (0, 0) for t in tiles), \
        "a tile reported zero offset on a scene with a planted 5-cell shift"


def test_a_cellular_scene_is_identifiable_whole_domain_too():
    """Confirms 2025-08-01 failed on CELLULARITY, not on method: the same
    whole-domain measurement that gave peak/zero 1.017 there gives >2 here."""
    from nowcast_data.alignment_gate import measure_scene
    tir, rain, lats, lons = _cellular_scene(shift=5)
    so = measure_scene(tir, rain, lats, lons, scene="whole")
    assert (so.dy, so.dx) == (-5, -5)
    assert so.identifiable and so.peak / so.zero > 1.5


def test_tiles_within_a_scene_are_clustered_not_pooled():
    """Tiles share their scene's storm field, scan time and registration, so
    the constant must come from SCENE-level values. Two scenes with planted
    5 px and 6 px displacements: sd(u) must be 0.707, the sd of {5, 6}, and
    the constant's SE must reflect n=2, not n=50."""
    import numpy as np
    from nowcast_data.alignment_gate import measure_scene_tiled, run_gate_tiled
    st = {}
    for i, (sh, sd) in enumerate([(5, 5), (6, 6)]):
        tir, rain, lats, lons = _cellular_scene(shift=sh, seed=sd)
        st[f"s{i}"] = measure_scene_tiled(tir, rain, lats, lons, scene=f"s{i}")
    r = run_gate_tiled(st)
    assert r.n_scenes_used == 2 and r.n_tiles_used >= 40
    assert abs(r.scene_level_sd_px - 0.707) < 0.05, r.scene_level_sd_px
    assert abs(r.constant_se_px - 0.5) < 0.05, r.constant_se_px
    assert r.verdict == "FAIL"          # a planted 5.5 px offset must fail
    assert any("NOT" in x and "tiles" in x for x in r.reasons)


def test_one_scene_cannot_establish_the_constant_however_many_tiles():
    """The scene-level term is common to every tile, so tiling cannot
    separate it. This must be INCONCLUSIVE, not a confident answer."""
    from nowcast_data.alignment_gate import measure_scene_tiled, run_gate_tiled
    tir, rain, lats, lons = _cellular_scene(shift=5)
    r = run_gate_tiled({"only": measure_scene_tiled(tir, rain, lats, lons)})
    assert r.verdict == "INCONCLUSIVE"
    assert "tiles do not help it" in " ".join(r.reasons)


# ---------------------------------------------------------------------------
# LUT clamp: the fill count is not the whole clamp
# ---------------------------------------------------------------------------

def test_lut_clamp_masks_the_whole_saturated_plateau():
    """Masking count 1023 removes the off-disk fill and leaves the rest of
    the LUT's flat tail decoding as a real, coldest-in-scene measurement.
    Measured on a real 3RIMG scan: TIR1 35,489 cells (0.449%), TIR2 23,312,
    MIR 89,929 (1.139%)."""
    import numpy as np
    from nowcast_data.insat import lut_clamp_value, mask_lut_clamp
    # A warm plateau is added so the fixture matches a real LUT, which
    # saturates at BOTH ends -- the first version of this test had only the
    # cold one and so could not have caught the mirrored defect.
    table = np.concatenate([np.full(50, 300.0),
                            np.linspace(299.0, 180.0, 850),
                            np.full(124, 179.7)])
    counts = np.array([[60, 500, 950, 1000, 1023]])
    bt = table[counts]
    assert lut_clamp_value(table) == 179.7
    out = mask_lut_clamp(bt, table)
    assert np.isfinite(out[0, :2]).all(), "real measurements survive"
    assert not np.isfinite(out[0, 2:]).any(), "the whole cold plateau is masked"


def test_clamp_value_is_read_per_channel_not_hardcoded():
    """TIR1 clamps at 179.86, TIR2 at 179.93, WV and MIR at 179.69. A single
    constant would be wrong for three of the four."""
    import numpy as np
    from nowcast_data.insat import lut_clamp_value
    for v in (179.86, 179.93, 179.69):
        t = np.concatenate([np.linspace(300.0, v + 1, 900), np.full(124, v)])
        assert lut_clamp_value(t) == v


def test_day_and_night_warm_bounds_differ():
    """The night bounds were calibrated at 23:30 UTC and fail every daylight
    scan -- the warm-end failure tracked solar elevation exactly."""
    from nowcast_data.insat import (DAY_SUN_ELEVATION_DEG,
                                    EXPECTED_BT_PERCENTILES,
                                    EXPECTED_BT_PERCENTILES_DAY)
    night = EXPECTED_BT_PERCENTILES["TIR1"][99.9]
    day = EXPECTED_BT_PERCENTILES_DAY["TIR1"][99.9]
    assert day[1] > night[1], "daytime land heating raises the warm tail"
    # MIR is looser again: at 3.9 um the signal carries reflected solar, so
    # its daytime BT is not a temperature at all.
    assert EXPECTED_BT_PERCENTILES_DAY["MIR"][99.9][1] > day[1]
    assert 0 < DAY_SUN_ELEVATION_DEG < 15


def test_lut_clamp_masks_BOTH_ends():
    """The table saturates warm as well as cold. Masking only the cold end
    left the mirrored case open -- MIR reserves 232 counts for 339.79 K."""
    import numpy as np
    from nowcast_data.insat import lut_plateaus, mask_lut_clamp
    t = np.concatenate([np.full(200, 340.0), np.linspace(339.0, 180.5, 700),
                        np.full(124, 179.7)])
    assert lut_plateaus(t) == (179.7, 340.0)
    out = mask_lut_clamp(np.array([[340.0, 300.0, 250.0, 179.7]]), t)
    assert np.isnan(out[0, 0]), "warm plateau masked"
    assert np.isnan(out[0, 3]), "cold plateau masked"
    assert out[0, 1] == 300.0 and out[0, 2] == 250.0, "real values untouched"


def test_wide_lut_steps_are_quantisation_not_sentinels():
    """All six --strict-scan failures were this. Near the cold end the MIR
    table moves 7-16.7 K per count, so one count's cells land on a value
    whose neighbours are 7-16 K away and the isolation score reads it as a
    spike. A true clamp is one value shared by MANY counts; quantisation is
    one count standing alone."""
    import numpy as np
    from nowcast_train.insat_ingest import QUANTISATION_STEP_K, _near
    coarse = np.array([185.98, 193.73, 197.16])
    assert _near(193.73, coarse)
    assert not _near(298.0, coarse), "the genuine 298 K warm mode is not flagged"
    assert QUANTISATION_STEP_K < 7.0, "must catch the 7-16.7 K/count region"
    assert QUANTISATION_STEP_K > 0.146, "must NOT catch the 0.146 K/count mode"
