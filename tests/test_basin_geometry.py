"""Basin geometry in the harness: "Gap 2" from the original audit.

The flood head scores on sub-basin polygons, not on the grid and not on
gauges. The array rank happens to match point geometry, so it would have run
today declared as point -- and produced a wrong number quietly, because
point geometry counts every element once. That is right for gauges (a gauge
is one observation) and wrong for basins (a basin is a variable amount of
land).
"""
import numpy as np
import pytest

from nowcast_eval import EvalConfig, evaluate
from nowcast_eval.contingency import contingency


def test_basin_geometry_is_accepted_and_recorded():
    cfg = EvalConfig(geometry="basin")
    pred = np.zeros((2, 3, 5)); obs = np.zeros((2, 3, 5))
    r = evaluate(pred, obs, cfg, weights=np.ones(5))
    assert r.pooled["geometry"] == "basin"
    assert r.pooled["basin_weighting"] == "area"
    assert "basin" in r.summary_table()


def test_basin_geometry_rejects_a_grid_shaped_array():
    with pytest.raises(ValueError, match=r"expects \(N, L, B\)"):
        evaluate(np.zeros((2, 3, 4, 4)), np.zeros((2, 3, 4, 4)),
                 EvalConfig(geometry="basin"), weights=np.ones(4))


def test_fss_is_skipped_for_basins():
    """Basin adjacency is a river network, not a raster. A neighbourhood
    filter over basin INDEX would be meaningless in the same way station
    order is."""
    r = evaluate(np.zeros((2, 3, 5)), np.zeros((2, 3, 5)),
                 EvalConfig(geometry="basin"), weights=np.ones(5))
    assert r.pooled["fss"] == []
    assert "FSS omitted" in r.summary_table()


def test_weights_are_required_and_the_error_says_why():
    """Silently defaulting to equal weights would reproduce the exact error
    the geometry field exists to prevent."""
    with pytest.raises(ValueError, match="orders of magnitude"):
        evaluate(np.zeros((2, 3, 5)), np.zeros((2, 3, 5)),
                 EvalConfig(geometry="basin"))


def test_equal_weighting_must_be_asked_for_explicitly():
    r = evaluate(np.zeros((2, 3, 5)), np.zeros((2, 3, 5)),
                 EvalConfig(geometry="basin", basin_weighting="equal"))
    assert r.pooled["basin_weighting"] == "equal"
    assert r.pooled["headline"]["weighted"] is False


def test_weights_are_rejected_for_grid_and_point():
    for g, shape in (("grid", (2, 3, 4, 4)), ("point", (2, 3, 5))):
        with pytest.raises(ValueError, match="only meaningful"):
            evaluate(np.zeros(shape), np.zeros(shape), EvalConfig(geometry=g),
                     weights=np.ones(shape[-1]))


def test_area_weighting_changes_the_score_it_does_not_just_relabel_it():
    """THE point of the whole exercise, as a number.

    Five basins. The forecast gets the four tiny headwaters right and misses
    the one large valley. Counted per basin that is 80% correct; counted per
    km^2 it is a bad forecast, because the valley is most of the land.
    """
    areas = np.array([5.0, 5.0, 5.0, 5.0, 500.0])
    obs = np.array([[[1.0, 1.0, 1.0, 1.0, 1.0]]])
    pred = np.array([[[1.0, 1.0, 1.0, 1.0, 0.0]]])      # misses the big one

    equal = evaluate(pred, obs, EvalConfig(geometry="basin",
                                           basin_weighting="equal"))
    area = evaluate(pred, obs, EvalConfig(geometry="basin"), weights=areas)

    pod_equal = equal.pooled["headline"]["pod"]
    pod_area = area.pooled["headline"]["pod"]
    assert np.isclose(pod_equal, 4 / 5)                 # 80% of basins
    assert np.isclose(pod_area, 20 / 520)               # 3.8% of the land
    assert pod_equal > 20 * pod_area, (pod_equal, pod_area)


def test_the_silent_failure_declaring_basins_as_point_geometry():
    """Basins declared as point RUN, and report a materially better score.

    This is the trap in its exact form: same arrays, same metric, no error,
    no warning -- a POD 21x too high, and provenance saying 'station
    locations'.
    """
    areas = np.array([5.0, 5.0, 5.0, 5.0, 500.0])
    obs = np.array([[[1.0, 1.0, 1.0, 1.0, 1.0]]])
    pred = np.array([[[1.0, 1.0, 1.0, 1.0, 0.0]]])

    as_point = evaluate(pred, obs, EvalConfig(geometry="point"))
    as_basin = evaluate(pred, obs, EvalConfig(geometry="basin"), weights=areas)

    assert as_point.pooled["geometry"] == "point"       # wrong provenance
    assert as_point.pooled["headline"]["pod"] > \
           20 * as_basin.pooled["headline"]["pod"]


def test_weighted_table_reduces_to_the_unweighted_one():
    """Uniform weights must reproduce counting exactly, or the weighted path
    is a second implementation rather than a generalisation."""
    rng = np.random.default_rng(3)
    pred = rng.uniform(size=(4, 30)); obs = (rng.uniform(size=(4, 30)) < 0.3)
    a = contingency(pred, obs, 0.5)
    b = contingency(pred, obs, 0.5, weights=np.ones((4, 30)))
    for k in ("hits", "false_alarms", "misses", "correct_negatives"):
        assert getattr(a, k) == getattr(b, k)
    assert np.isclose(a.csi, b.csi) and np.isclose(a.sedi, b.sedi, equal_nan=True)
    assert b.weighted and not a.weighted


def test_scaling_all_weights_leaves_every_ratio_unchanged():
    """Every metric is a ratio, so the counting measure's units cancel --
    km^2 or m^2 must give the same skill."""
    rng = np.random.default_rng(4)
    pred = rng.uniform(size=(4, 30)); obs = (rng.uniform(size=(4, 30)) < 0.3)
    w = rng.uniform(1, 500, size=(4, 30))
    a = contingency(pred, obs, 0.5, weights=w)
    b = contingency(pred, obs, 0.5, weights=w * 1e6)
    for m in ("pod", "far", "csi", "ets", "hss", "sedi", "base_rate"):
        assert np.isclose(getattr(a, m), getattr(b, m), equal_nan=True), m


def test_negative_weights_are_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        contingency(np.zeros((2, 2)), np.zeros((2, 2)), 0.5,
                    weights=np.array([[1.0, -1.0], [1.0, 1.0]]))


def test_masked_basins_are_dropped_not_counted_as_dry():
    pred = np.array([[[0.9, 0.9, 0.9]]])
    obs = np.array([[[1.0, 1.0, 0.0]]])
    mask = np.array([True, True, False])
    r = evaluate(pred, obs, EvalConfig(geometry="basin"),
                 mask=mask, weights=np.array([10.0, 10.0, 1000.0]))
    h = r.pooled["headline"]
    assert h["false_alarms"] == 0        # the masked big basin is not an FA
    assert h["hits"] == 20.0


def test_multihazard_accepts_a_flood_head_on_basin_geometry():
    """The three heads genuinely have three geometries, and evaluate_multi
    already takes per-hazard configs -- confirm end to end."""
    from nowcast_eval.multihazard import evaluate_multi
    rng = np.random.default_rng(5)
    preds = {"thunderstorm": rng.uniform(size=(2, 3, 8, 8)),
             "cloudburst": rng.uniform(size=(2, 3, 12)),
             "flash_flood": rng.uniform(size=(2, 3, 5))}
    obs = {k: (rng.uniform(size=v.shape) < 0.3).astype(float)
           for k, v in preds.items()}
    cfgs = {"thunderstorm": EvalConfig(geometry="grid"),
            "cloudburst": EvalConfig(geometry="point"),
            "flash_flood": EvalConfig(geometry="basin")}
    res = evaluate_multi(preds, obs, cfgs,
                         weights={"flash_flood": np.array([5., 5., 5., 5., 500.])})
    assert res["flash_flood"].pooled["geometry"] == "basin"
    assert res["thunderstorm"].pooled["fss"]          # grid keeps FSS
    assert not res["flash_flood"].pooled["fss"]
    table = res.summary_table()
    assert "flash_flood" in table and "MULTI-HAZARD" in table
