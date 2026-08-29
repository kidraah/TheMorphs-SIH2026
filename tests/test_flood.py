"""Flood routing: known answers, and the traps that produce plausible lies.

Every number here is checkable by hand or fixed by a conservation law. The
routing is deterministic physics, so a test that merely runs it proves
nothing -- what matters is the cases where a wrong implementation still
returns a confident, reasonable-looking number.
"""
import numpy as np
import pytest

from nowcast_flood import (FloodRouter, aggregate, adjust_amc, build_flow_grid,
                           curve_number, delineate, flood_exposure,
                           flow_accumulate, peak_discharge, potential_retention,
                           properties, runoff_depth, time_of_concentration,
                           time_to_peak, verify_against_upa)
from nowcast_flood.timing import KIRPICH_MAX_KM2

# A Y-shaped network: three headwater columns joining at (3,1), then a trunk.
Y_DIR = np.array([[4, 4, 4],
                  [4, 4, 4],
                  [2, 4, 8],
                  [1, 4, 16],
                  [1, 4, 16],
                  [1, 0, 16]])


def y_grid():
    fg = build_flow_grid(Y_DIR)
    return fg, flow_accumulate(fg)


# ---------------------------------------------------------------------------
# Curve number
# ---------------------------------------------------------------------------

def test_runoff_is_zero_below_initial_abstraction():
    """THE trap. (P - 0.2S)^2 is positive on both sides of the threshold, so
    an unguarded implementation invents runoff from rain that produces none
    -- and invents MORE of it the smaller the storm."""
    cn = np.array([36.0])                       # forest, S = 451.6 mm
    s = float(potential_retention(cn)[0])
    naive = (1.0 - 0.2 * s) ** 2 / (1.0 + 0.8 * s)
    assert naive > 20, "the trap should be large here, or the test proves nothing"
    assert runoff_depth(np.array([1.0]), cn)[0] == 0.0
    # and still zero right up to the threshold
    assert runoff_depth(np.array([0.2 * s - 1e-9]), cn)[0] == 0.0
    assert runoff_depth(np.array([0.2 * s + 10.0]), cn)[0] > 0.0


def test_impervious_surface_runs_off_completely():
    """CN 100 -> S = 0 -> Q = P exactly. An analytic anchor, not a tolerance."""
    p = np.array([0.0, 1.0, 50.0, 300.0])
    assert np.allclose(runoff_depth(p, np.full(4, 100.0)), p)


def test_runoff_never_exceeds_rainfall():
    """Conservation. Violating it would be invisible in any single number."""
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 400, 5000)
    cn = rng.uniform(30, 100, 5000)
    q = runoff_depth(p, cn)
    assert np.all(q <= p + 1e-9)
    assert np.all(q >= 0)


def test_runoff_is_monotonic_in_rain_and_in_cn():
    p = np.linspace(0, 300, 200)
    q = runoff_depth(p, np.full(200, 70.0))
    assert np.all(np.diff(q) >= -1e-12)
    cn = np.linspace(31, 100, 200)
    q2 = runoff_depth(np.full(200, 80.0), cn)
    assert np.all(np.diff(q2) >= -1e-12)


def test_amc_conversions_are_the_published_ones():
    assert adjust_amc(np.array([70.0]), "II")[0] == 70.0
    assert np.isclose(adjust_amc(np.array([70.0]), "I")[0], 49.5, atol=0.1)
    assert np.isclose(adjust_amc(np.array([70.0]), "III")[0], 84.3, atol=0.1)
    with pytest.raises(ValueError):
        adjust_amc(np.array([70.0]), "IV")


def test_antecedent_moisture_dominates_the_answer():
    """Recorded because it justifies AMC being a required, stated choice:
    the same storm on the same land gives 2.8 mm dry and 42.2 mm wet."""
    dry, wet = (float(runoff_depth(np.array([80.0]),
                                   adjust_amc(np.array([70.0]), c))[0])
                for c in ("I", "III"))
    assert wet / dry > 10, (dry, wet)


def test_unmapped_landcover_is_nan_not_a_guess():
    cn = curve_number(np.array([[10, 999]]), np.array([[2, 2]]))
    assert cn[0, 0] == 60 and np.isnan(cn[0, 1])


def test_drained_soil_groups_map_to_their_parent_group():
    """HYSOGs250m codes 11..14 are drained A..D. Dropping them would blank
    much of the Indo-Gangetic plain."""
    assert curve_number(np.array([[40]]), np.array([[13]]))[0, 0] == \
           curve_number(np.array([[40]]), np.array([[3]]))[0, 0]


# ---------------------------------------------------------------------------
# Flow routing
# ---------------------------------------------------------------------------

def test_accumulation_conserves_area():
    """Every cell reaches a terminal, so terminal accumulation sums to N."""
    fg, upa = y_grid()
    assert upa[fg.is_terminal.reshape(fg.shape)].sum() == fg.n


def test_accumulation_matches_hand_computation():
    _, upa = y_grid()
    assert upa[5, 1] == 18                 # the mouth drains everything
    # 3 columns of 3 from above, plus the two lateral inflows on row 3,
    # plus the cell itself. Getting this wrong by hand is exactly why the
    # conservation test above is the one that carries the weight.
    assert upa[3, 1] == 12
    assert upa[0, 0] == 1                  # a headwater is only itself


def test_off_grid_flow_truncates_and_does_not_wrap():
    """Wrapping would connect opposite edges of India and the resulting
    basin would look entirely ordinary."""
    d = np.array([[16, 16], [16, 16]])     # everything flows west, off-grid
    fg = build_flow_grid(d)
    assert fg.downstream[0] == -1 and fg.downstream[2] == -1
    assert flow_accumulate(fg).sum() == 6  # 1+2 per row


def test_cyclic_direction_field_is_rejected():
    """A cycle means the encoding is wrong; routing it anyway would produce
    numbers rather than an error."""
    d = np.array([[1, 4], [64, 16]])       # a 2x2 loop
    with pytest.raises(ValueError, match="cycle"):
        flow_accumulate(build_flow_grid(d))


def test_verify_against_upa_catches_a_transposed_convention():
    """The gate that stands in for having no real `dir` tile yet: a wrong
    convention still routes and still returns plausible basins."""
    fg, upa = y_grid()
    # exclude_boundary=False: on a 6x3 toy every cell touches an edge.
    assert verify_against_upa(fg, upa, 1.0, exclude_boundary=False)["passed"]
    # transposing the direction codes is a realistic misreading
    wrong = build_flow_grid(Y_DIR.T)
    res = verify_against_upa(wrong, upa.T, 1.0, exclude_boundary=False)
    assert not res["passed"], res


def test_boundary_contaminated_cells_are_excluded_by_default():
    """MERIT's upa is global, so at a tile edge it counts area outside the
    tile. Comparing those cells would report a disagreement caused by the
    crop -- which looks exactly like a wrong D8 convention."""
    from nowcast_flood.flow import boundary_contaminated
    fg, _ = y_grid()
    flag = boundary_contaminated(fg)
    assert flag.all()                      # every cell of a 6x3 touches an edge
    # An interior catchment that touches no edge: the border is all terminal,
    # and a 5x5 interior block drains east then south to (5, 5).
    d = np.zeros((7, 7), dtype=int)
    d[1:6, 1:5] = 1                        # east
    d[1:5, 5] = 4                          # south
    fg2 = build_flow_grid(d)
    f2 = boundary_contaminated(fg2)
    assert f2[0, 0] and f2[:, 0].all() and f2[-1, :].all()   # the border itself
    assert not f2[1, 1], "an interior headwater is not contaminated"
    assert not f2[5, 5], "an interior outlet fed only by interior cells is clean"
    assert flow_accumulate(fg2)[5, 5] == 25


# ---------------------------------------------------------------------------
# Basins
# ---------------------------------------------------------------------------

def test_delineation_partitions_every_cell_exactly_once():
    fg, upa = y_grid()
    lab = delineate(fg, upa, channel_km2=3.0)
    p = properties(fg, lab, 1.0, 100.0, upa, channel_km2=3.0)
    assert p.area_km2.sum() == fg.n
    assert set(np.unique(lab)) == set(p.ids)


def test_confluence_creates_a_new_reach():
    fg, upa = y_grid()
    lab = delineate(fg, upa, channel_km2=3.0)
    assert len(np.unique(lab)) == 4        # three headwaters + one trunk
    assert lab[5, 1] == lab[3, 1]          # trunk is one reach through the mouth
    assert lab[0, 0] != lab[0, 2]          # separate headwaters


def test_diagonal_channel_steps_are_longer():
    """Ignoring this underestimates length by up to 41%, and Kirpich takes
    L^0.77, so it lands straight in the arrival time."""
    fg, upa = y_grid()
    lab = delineate(fg, upa, channel_km2=3.0)
    p = properties(fg, lab, 1.0, 100.0, upa, channel_km2=3.0)
    lens = sorted(p.channel_length_m)
    assert np.isclose(lens[0], 100.0)                    # one cardinal cell
    assert np.isclose(lens[1], 100 * np.sqrt(2))         # one diagonal cell


def test_single_cell_reaches_get_a_real_slope():
    """Measuring the drop WITHIN a reach gives every single-cell reach a
    slope of exactly zero, hence an undefined arrival time -- at the channel
    threshold that is a large and artificial share of the headwaters."""
    fg, upa = y_grid()
    lab = delineate(fg, upa, channel_km2=3.0)
    elev = np.linspace(300, 0, 18).reshape(6, 3)
    p = properties(fg, lab, 16.0, 4000.0, upa, elev, channel_km2=3.0)
    assert np.all(np.isfinite(p.channel_slope)), p.channel_slope
    assert np.all(p.channel_slope > 0)


def test_aggregate_is_area_weighted_not_cell_counted():
    fg, upa = y_grid()
    lab = delineate(fg, upa, channel_km2=3.0)
    p = properties(fg, lab, 1.0, 100.0, upa, channel_km2=3.0)
    field = np.zeros((6, 3)); field[3:, :] = 10.0     # the trunk basin only
    means = aggregate(field, p, "mean")
    trunk = p.ids.tolist().index(lab[5, 1])
    assert np.isclose(means[trunk], 10.0)
    assert np.allclose(np.delete(means, trunk), 0.0)
    assert np.isclose(aggregate(field, p, "sum")[trunk], 90.0)


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------

def test_kirpich_metric_constant_matches_the_imperial_original():
    """0.0078 L_ft^0.77 S^-0.385 == 0.0195 L_m^0.77 S^-0.385."""
    L_m, S = 12000.0, 0.015
    imperial = 0.0078 * (L_m / 0.3048) ** 0.77 * S ** -0.385
    metric = float(time_of_concentration([L_m], [S], method="kirpich").t_c_min[0])
    assert np.isclose(imperial, metric, rtol=1e-12), (imperial, metric)


def test_zero_slope_is_nan_not_a_huge_number():
    """A very large t_c sorts to the end of a warning list and reads as a
    calm basin."""
    for m in ("kirpich", "watt_chow"):
        assert np.isnan(time_of_concentration([1000.0], [0.0], method=m).t_c_min[0])


def test_out_of_range_basins_are_flagged():
    """Kirpich was fitted on 0.004-0.45 km^2. Our basins start at ~25 km^2."""
    t = time_of_concentration([12000.0], [0.015], [50.0], method="kirpich")
    assert not t.within_fitted_range[0]
    assert "outside kirpich" in t.report()
    t_ok = time_of_concentration([300.0], [0.05], [KIRPICH_MAX_KM2 / 2],
                                 method="kirpich")
    assert t_ok.within_fitted_range[0]
    # ...and the default method IS in range at the same 50 km^2, which is
    # the entire reason it is the default.
    assert time_of_concentration([12000.0], [0.015], [50.0]).within_fitted_range[0]


def test_steeper_and_shorter_basins_respond_sooner():
    tc = time_of_concentration([12000.0, 12000.0, 3000.0],
                               [0.005, 0.05, 0.005], method="kirpich").t_c_min
    assert tc[1] < tc[0] and tc[2] < tc[0]


def test_time_to_peak_is_not_time_of_concentration():
    """They were once the same field, and report() printed t_c under the
    label 'time to peak' -- 136 min where the truth was 111."""
    tc = np.array([135.9])
    assert np.isclose(time_to_peak(tc, 60.0)[0], 30.0 + 0.6 * 135.9)
    t = time_of_concentration([12000.0], [0.015], rain_duration_min=60.0,
                              method="kirpich")
    assert not np.isclose(t.t_c_min[0], t.t_peak_min[0])


def test_peak_discharge_scales_as_the_unit_hydrograph_says():
    """q_p = 0.208 A Q / T_p: linear in area and in runoff depth."""
    q1 = peak_discharge([40.0], [50.0], [120.0])[0]
    assert np.isclose(peak_discharge([40.0], [100.0], [120.0])[0], 2 * q1)
    assert np.isclose(peak_discharge([80.0], [50.0], [120.0])[0], 2 * q1)
    assert np.isnan(peak_discharge([40.0], [50.0], [np.nan])[0])


# ---------------------------------------------------------------------------
# Exposure and the full interface
# ---------------------------------------------------------------------------

def test_exposure_needs_low_ground_AND_upstream_water():
    fg, upa = y_grid()
    lab = delineate(fg, upa, channel_km2=3.0)
    p = properties(fg, lab, 1.0, 100.0, upa, channel_km2=3.0)
    hand = np.where(np.arange(18).reshape(6, 3) < 9, 1.0, 50.0)
    wet = np.zeros(p.n); wet[p.ids.tolist().index(lab[0, 0])] = 100.0
    e = flood_exposure(hand, wet, p, 5.0, 25.0)
    assert e.exposed[0, 0] and not e.exposed[0, 1]      # dry basin, low ground
    assert not e.exposed[5, 0]                          # wet basin, high ground
    assert "no settlement layer" in e.report()


def test_population_weighting_changes_which_basin_ranks_first():
    """Area-only ranking puts empty headwaters above cities."""
    fg, upa = y_grid()
    lab = delineate(fg, upa, channel_km2=3.0)
    p = properties(fg, lab, 1.0, 100.0, upa, channel_km2=3.0)
    hand = np.zeros((6, 3))
    wet = np.full(p.n, 100.0)
    pop = np.zeros((6, 3)); pop[5, 1] = 1e6            # one populated cell
    e = flood_exposure(hand, wet, p, 5.0, 25.0, population=pop)
    trunk = p.ids.tolist().index(lab[5, 1])
    assert np.argmax(e.area_km2) == trunk or e.area_km2[trunk] > 0
    assert np.argmax(e.people) == trunk
    assert e.people.sum() == 1e6


def test_route_returns_the_three_promised_quantities():
    elev = np.linspace(300, 0, 18).reshape(6, 3)
    _, upa = y_grid()
    r = FloodRouter.from_layers(direction=Y_DIR, upa_km2=upa,
                                curve_number=np.full((6, 3), 75.0),
                                hand_m=np.full((6, 3), 2.0), elevation_m=elev,
                                cell_area_km2=16.0, cell_size_m=4000.0,
                                channel_km2=3.0, amc="III")
    fc = r.route(np.full((6, 3), 90.0), rain_duration_min=60)
    assert fc.n_basins == 4
    for arr in (fc.basin_risk, fc.discharge_m3s, fc.time_to_arrival_min):
        assert arr.shape == (4,) and np.all(np.isfinite(arr))
    assert np.all((fc.basin_risk >= 0) & (fc.basin_risk <= 1))
    assert fc.meta["amc"] == "III"                # the choice is recorded
    grid = fc.to_grid(r.basins.labels)
    assert grid.shape == (6, 3) and np.all(np.isfinite(grid))


def test_no_rain_produces_no_flood():
    _, upa = y_grid()
    r = FloodRouter.from_layers(direction=Y_DIR, upa_km2=upa,
                                curve_number=np.full((6, 3), 75.0),
                                elevation_m=np.linspace(300, 0, 18).reshape(6, 3),
                                cell_area_km2=16.0, cell_size_m=4000.0,
                                channel_km2=3.0)
    fc = r.route(np.zeros((6, 3)))
    assert np.allclose(fc.runoff_mm, 0.0)
    assert np.allclose(fc.discharge_m3s, 0.0)
    assert np.allclose(fc.basin_risk, 0.0)


def test_risk_reference_is_explicit_not_the_fields_own_maximum():
    """Normalising by the current field's max puts a 1.0 in every forecast,
    including a dry day."""
    _, upa = y_grid()
    r = FloodRouter.from_layers(direction=Y_DIR, upa_km2=upa,
                                curve_number=np.full((6, 3), 75.0),
                                elevation_m=np.linspace(300, 0, 18).reshape(6, 3),
                                cell_area_km2=16.0, cell_size_m=4000.0,
                                channel_km2=3.0)
    drizzle = r.route(np.full((6, 3), 20.0), risk_reference_m3s=5000.0)
    assert drizzle.basin_risk.max() < 1.0
    assert drizzle.meta["risk_reference_m3s"] == 5000.0


def test_rain_grid_must_match_the_terrain():
    _, upa = y_grid()
    r = FloodRouter.from_layers(direction=Y_DIR, upa_km2=upa,
                                curve_number=np.full((6, 3), 75.0),
                                cell_area_km2=16.0, cell_size_m=4000.0,
                                channel_km2=3.0)
    with pytest.raises(ValueError, match="does not match"):
        r.route(np.zeros((5, 5)))


# ---------------------------------------------------------------------------
# Antecedent moisture (AMC as an input, not an assumption)
# ---------------------------------------------------------------------------

def test_antecedent_window_excludes_the_day_itself():
    """Including the current day leaks the event into its own antecedent
    condition: a storm raises its own CN and inflates its own runoff. It
    would improve every hindcast and be unavailable at forecast time."""
    from nowcast_flood import antecedent_5day
    daily = np.arange(10, dtype=float)[:, None]
    a = antecedent_5day(daily)[:, 0]
    assert np.all(np.isnan(a[:5]))          # not enough history
    assert a[5] == 0 + 1 + 2 + 3 + 4
    assert a[6] == 1 + 2 + 3 + 4 + 5


def test_antecedent_needs_enough_history():
    from nowcast_flood import antecedent_5day
    with pytest.raises(ValueError, match="at least 6 days"):
        antecedent_5day(np.zeros((3, 2)))


def test_amc_classes_match_the_nrcs_thresholds():
    from nowcast_flood import AMC_THRESHOLDS, amc_class
    lo, hi = AMC_THRESHOLDS["growing"]
    assert amc_class(np.array([lo - 0.1]))[0] == 1
    assert amc_class(np.array([lo + 0.1]))[0] == 2
    assert amc_class(np.array([hi + 0.1]))[0] == 3
    assert amc_class(np.array([5.0]), "dormant")[0] == 1   # dry in dormant too
    assert amc_class(np.array([20.0]), "dormant")[0] == 2  # but average in dormant


def test_continuous_amc_removes_the_threshold_cliff():
    """0.2 mm of rain five days ago must not move published risk by 20 CN."""
    from nowcast_flood import curve_number_from_antecedent as cn_ant
    cn2 = np.array([70.0])
    step = [float(cn_ant(cn2, np.array([p]), continuous=False)[0])
            for p in (35.5, 35.7)]
    cont = [float(cn_ant(cn2, np.array([p]))[0]) for p in (35.5, 35.7)]
    assert step[1] - step[0] > 20            # the cliff is real
    assert cont[1] - cont[0] < 1             # and the interpolation removes it


def test_continuous_amc_reproduces_the_classes_at_the_anchors():
    from nowcast_flood import AMC_THRESHOLDS, adjust_amc
    from nowcast_flood import curve_number_from_antecedent as cn_ant
    cn2 = np.array([70.0])
    lo, hi = AMC_THRESHOLDS["growing"]
    assert np.isclose(cn_ant(cn2, np.array([lo]))[0], adjust_amc(cn2, "I")[0])
    assert np.isclose(cn_ant(cn2, np.array([hi]))[0], adjust_amc(cn2, "III")[0])
    assert np.isclose(cn_ant(cn2, np.array([(lo + hi) / 2]))[0], 70.0)


def test_antecedent_cn_is_monotonic():
    from nowcast_flood import curve_number_from_antecedent as cn_ant
    p = np.linspace(0, 120, 200)
    cn = cn_ant(np.full(200, 70.0), p)
    assert np.all(np.diff(cn) >= -1e-9)


# ---------------------------------------------------------------------------
# Time-of-concentration method choice
# ---------------------------------------------------------------------------

def test_default_method_is_fitted_at_our_basin_sizes():
    """Kirpich's range stops at 0.45 km^2; our basins start around 25."""
    from nowcast_flood.timing import DEFAULT_METHOD, FITTED_RANGE_KM2
    lo, hi = FITTED_RANGE_KM2[DEFAULT_METHOD]
    assert lo <= 25.0 <= hi and hi >= 1000.0
    assert FITTED_RANGE_KM2["kirpich"][1] < 1.0


def test_watt_chow_matches_its_published_form():
    """t_c [h] = 0.000326 (L / sqrt(S))^0.79, L in m."""
    L, S = 12000.0, 0.015
    expect = 60.0 * 0.000326 * (L / np.sqrt(S)) ** 0.79
    got = float(time_of_concentration([L], [S], method="watt_chow").t_c_min[0])
    assert np.isclose(expect, got, rtol=1e-12)


def test_giandotti_needs_relief_and_says_so_by_returning_nan():
    t = time_of_concentration([12000.0], [0.015], [50.0], method="giandotti")
    assert np.isnan(t.t_c_min[0])
    t2 = time_of_concentration([12000.0], [0.015], [50.0], method="giandotti",
                               mean_elev_above_outlet_m=[180.0])
    assert np.isfinite(t2.t_c_min[0])


def test_methods_disagree_and_the_spread_is_reported():
    """Three regressions disagreeing by a factor is information the operator
    needs; one number to the minute is not."""
    from nowcast_flood import compare_methods
    c = compare_methods([12000.0], [0.015], [50.0], [180.0], 60.0)
    assert c["spread_ratio"][0] > 1.3
    assert c["t_peak_min_low"][0] < c["t_peak_min_high"][0]
    assert set(c["methods"]) == {"kirpich", "watt_chow", "giandotti"}


def test_unknown_method_is_rejected():
    with pytest.raises(ValueError, match="method must be"):
        time_of_concentration([1000.0], [0.01], method="vibes")


def test_router_records_the_method_and_the_arrival_band():
    _, upa = y_grid()
    r = FloodRouter.from_layers(direction=Y_DIR, upa_km2=upa,
                                curve_number=np.full((6, 3), 75.0),
                                elevation_m=np.linspace(300, 0, 18).reshape(6, 3),
                                cell_area_km2=16.0, cell_size_m=4000.0,
                                channel_km2=3.0)
    fc = r.route(np.full((6, 3), 90.0), rain_duration_min=60)
    assert fc.meta["tc_method"] == "watt_chow"
    assert np.all(fc.arrival_low_min <= fc.time_to_arrival_min + 1e-9)
    assert np.all(fc.arrival_high_min >= fc.time_to_arrival_min - 1e-9)
