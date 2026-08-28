"""Known-answer tests: proof that the ruler is straight.

The harness cannot be validated by "the numbers look plausible" -- that is
exactly how a subtly wrong metric survives to the final slide. Instead we
feed it cases whose answers are known analytically and check for exact
equality. A perfect forecast must score exactly 1.000, not 0.997.
"""
import numpy as np
import pytest

from nowcast_eval import (EvalConfig, brier_score, contingency, evaluate, fss,
                          probabilistic_scores, reliability_curve)
from nowcast_eval.baselines import climatology, persistence
from nowcast_eval.fss import neighborhood_fractions


# --------------------------------------------------------------------------
# Contingency table: hand-computed
# --------------------------------------------------------------------------

def test_contingency_hand_computed():
    """A table worked out on paper, cell by cell."""
    #            obs=1  obs=1  obs=0  obs=0  obs=0
    pred = np.array([0.9, 0.1, 0.8, 0.2, 0.3])
    obs = np.array([1, 1, 0, 0, 0])
    t = contingency(pred, obs, threshold=0.5)

    assert t.hits == 1              # 0.9 & obs=1
    assert t.misses == 1            # 0.1 & obs=1
    assert t.false_alarms == 1      # 0.8 & obs=0
    assert t.correct_negatives == 2 # 0.2, 0.3 & obs=0
    assert t.n == 5

    assert t.pod == pytest.approx(1 / 2)
    assert t.far == pytest.approx(1 / 2)
    assert t.csi == pytest.approx(1 / 3)          # 1 / (1+1+1)
    assert t.success_ratio == pytest.approx(1 / 2)
    assert t.frequency_bias == pytest.approx(2 / 2)
    assert t.base_rate == pytest.approx(2 / 5)


def test_perfect_forecast_scores_exactly_one():
    rng = np.random.default_rng(0)
    obs = (rng.random((4, 3, 16, 16)) < 0.1).astype(float)
    t = contingency(obs, obs, threshold=0.5)
    assert t.pod == 1.0
    assert t.far == 0.0
    assert t.csi == 1.0
    assert t.ets == pytest.approx(1.0)
    assert t.hss == pytest.approx(1.0)
    assert t.frequency_bias == 1.0
    assert brier_score(obs, obs) == 0.0


def test_inverted_forecast_scores_exactly_zero():
    rng = np.random.default_rng(1)
    obs = (rng.random((2, 2, 12, 12)) < 0.2).astype(float)
    t = contingency(1.0 - obs, obs, threshold=0.5)
    assert t.hits == 0
    assert t.pod == 0.0
    assert t.csi == 0.0
    assert t.far == 1.0


def test_forecast_of_nothing():
    """The degenerate model that never fires. FAR is 0/0 -> nan, not 0."""
    obs = np.zeros((1, 1, 8, 8)); obs[0, 0, 3, 3] = 1
    t = contingency(np.zeros_like(obs), obs, threshold=0.5)
    assert t.pod == 0.0
    assert t.csi == 0.0
    assert np.isnan(t.far), "0 hits and 0 false alarms must be undefined, not 0.0"


def test_base_rate_is_reported():
    """Guards the design rule: CSI is meaningless without the base rate."""
    obs = np.zeros(10_000); obs[:3] = 1
    t = contingency(np.zeros_like(obs), obs, threshold=0.5)
    assert t.base_rate == pytest.approx(3e-4)


# --------------------------------------------------------------------------
# Masking
# --------------------------------------------------------------------------

def test_masked_cells_are_dropped_not_counted_as_no_event():
    pred = np.array([0.9, 0.9, 0.9, 0.9])
    obs = np.array([1, 0, 0, 0])
    mask = np.array([True, False, False, False])

    t = contingency(pred, obs, 0.5, mask=mask)
    assert (t.hits, t.false_alarms, t.misses, t.correct_negatives) == (1, 0, 0, 0)
    assert t.far == 0.0, "masked cells must not become false alarms"

    t_all = contingency(pred, obs, 0.5)
    assert t_all.far == pytest.approx(3 / 4)


def test_nan_predictions_are_excluded():
    pred = np.array([0.9, np.nan, 0.1])
    obs = np.array([1, 1, 0])
    t = contingency(pred, obs, 0.5)
    assert t.n == 2


# --------------------------------------------------------------------------
# FSS: exact analytic anchors
# --------------------------------------------------------------------------

def test_fss_identical_fields_is_exactly_one():
    rng = np.random.default_rng(2)
    f = (rng.random((3, 24, 24)) < 0.15).astype(float)
    for size in (1, 3, 9):
        assert fss(f, f, 0.5, size).fss == pytest.approx(1.0)


def test_fss_disjoint_fields_at_unit_neighborhood_is_exactly_zero():
    """Analytic: for disjoint binary fields with size=1,
       FBS = mean(pf) + mean(po) = FBS_worst, so FSS = 0 exactly."""
    obs = np.zeros((1, 10, 10)); obs[0, 2:4, 2:4] = 1
    pred = np.zeros((1, 10, 10)); pred[0, 7:9, 7:9] = 1
    assert fss(pred, obs, 0.5, size=1).fss == pytest.approx(0.0)


def test_fss_rewards_near_misses_as_neighborhood_grows():
    """The entire reason FSS exists: a displaced-but-correct forecast should
    recover skill at the scale of the displacement, while pixel CSI stays 0."""
    obs = np.zeros((1, 40, 40)); obs[0, 18:22, 10:14] = 1
    pred = np.zeros((1, 40, 40)); pred[0, 18:22, 16:20] = 1  # 6 px east

    assert contingency(pred, obs, 0.5).csi == 0.0, "no pixel overlap"

    scores = [fss(pred, obs, 0.5, size=s).fss for s in (1, 5, 11, 21, 31, 41)]
    assert scores[0] == pytest.approx(0.0), "no credit at pixel scale"
    assert all(b >= a - 1e-12 for a, b in zip(scores, scores[1:])), scores

    # Exact anchor rather than a magic number: once the window is wide enough
    # that every window spans both blobs, and the blobs contain the same number
    # of event pixels, the two fraction fields are identical -> FSS is exactly 1.
    assert fss(pred, obs, 0.5, size=61).fss == pytest.approx(1.0)


def test_fss_is_insensitive_to_domain_padding():
    """The same displacement on a 40x40 and a 200x200 domain must score the
    same at a moderate neighbourhood -- otherwise the metric is measuring the
    size of the grid rather than the quality of the forecast."""
    small_o = np.zeros((1, 40, 40)); small_o[0, 18:22, 10:14] = 1
    small_p = np.zeros((1, 40, 40)); small_p[0, 18:22, 16:20] = 1
    big_o = np.zeros((1, 200, 200)); big_o[0, 98:102, 90:94] = 1
    big_p = np.zeros((1, 200, 200)); big_p[0, 98:102, 96:100] = 1

    assert fss(small_p, small_o, 0.5, 11).fss == pytest.approx(
        fss(big_p, big_o, 0.5, 11).fss)


def test_fss_matches_its_own_definition():
    """Recompute 1 - FBS/FBS_worst by hand from the returned components."""
    rng = np.random.default_rng(3)
    obs = (rng.random((2, 20, 20)) < 0.2).astype(float)
    pred = rng.random((2, 20, 20))
    r = fss(pred, obs, 0.4, size=5)
    assert r.fss == pytest.approx(1.0 - r.fbs / r.fbs_worst)


def test_neighborhood_fractions_have_no_edge_bias():
    """A field that is 1 everywhere must give fraction 1.0 everywhere,
    including corners. Counting out-of-domain as no-event is the classic
    bug -- it would give 4/9 at a corner with a 3x3 window."""
    ones = np.ones((1, 6, 6))
    valid = np.ones_like(ones, dtype=bool)
    frac = neighborhood_fractions(ones.astype(bool), valid, size=3)
    assert np.allclose(frac, 1.0)
    assert frac[0, 0, 0] == pytest.approx(1.0)


def test_fss_masked_cells_excluded_from_fractions():
    field = np.ones((1, 5, 5))
    valid = np.ones((1, 5, 5), dtype=bool)
    valid[0, 0, 0] = False
    frac = neighborhood_fractions(field.astype(bool), valid, size=3)
    assert np.allclose(frac[valid], 1.0), "invalid cells must not dilute the fraction"


def test_fss_both_fields_empty_is_undefined():
    """A model that never fires must not score 1.0 on a quiet day."""
    z = np.zeros((1, 8, 8))
    assert np.isnan(fss(z, z, 0.5, size=3).fss)


# --------------------------------------------------------------------------
# Probabilistic
# --------------------------------------------------------------------------

def test_brier_hand_computed():
    pred = np.array([0.0, 1.0, 0.5, 0.25])
    obs = np.array([0, 1, 1, 0])
    expected = (0.0 + 0.0 + 0.25 + 0.0625) / 4
    assert brier_score(pred, obs) == pytest.approx(expected)


def test_climatology_has_exactly_zero_skill():
    """BSS is defined against climatology, so climatology must score 0.0.
    If this drifts, every BSS the project reports is wrong."""
    rng = np.random.default_rng(4)
    obs = (rng.random((5, 3, 16, 16)) < 0.07).astype(float)
    base = float(obs.mean())
    s = probabilistic_scores(climatology(base, obs.shape), obs)
    assert s.bss == pytest.approx(0.0, abs=1e-12)
    assert s.base_rate == pytest.approx(base)


def test_perfect_probabilities_have_skill_one():
    rng = np.random.default_rng(5)
    obs = (rng.random((2, 2, 10, 10)) < 0.2).astype(float)
    s = probabilistic_scores(obs, obs)
    assert s.brier == 0.0
    assert s.bss == pytest.approx(1.0)


def test_reliability_of_a_calibrated_forecast():
    """Build a forecast that is honest by construction: where it says 0.3,
    events occur 30% of the time. The curve must sit on the diagonal."""
    rng = np.random.default_rng(6)
    probs = rng.choice([0.05, 0.35, 0.65, 0.95], size=400_000)
    obs = (rng.random(probs.shape) < probs).astype(float)
    c = reliability_curve(probs, obs, n_bins=10)
    occupied = c.counts > 1000
    assert np.allclose(c.mean_forecast[occupied], c.observed_freq[occupied], atol=0.01)


def test_reliability_bins_account_for_every_sample():
    rng = np.random.default_rng(7)
    p = rng.random(5000)
    p[:10] = 1.0  # p == 1.0 must land in the last bin, not be dropped
    c = reliability_curve(p, (rng.random(5000) < 0.3).astype(float), n_bins=10)
    assert c.counts.sum() == 5000


# --------------------------------------------------------------------------
# End-to-end
# --------------------------------------------------------------------------

def _toy(n=6, lead=4, h=32, w=32, seed=0):
    rng = np.random.default_rng(seed)
    obs = (rng.random((n, lead, h, w)) < 0.05).astype(float)
    return obs


def test_evaluate_perfect_end_to_end():
    obs = _toy()
    r = evaluate(obs, obs, EvalConfig(name="perfect"))
    for l in r.per_lead:
        assert l.headline["csi"] == 1.0
        assert l.probabilistic["brier"] == 0.0
    assert r.pooled["headline"]["csi"] == 1.0


def test_evaluate_is_deterministic():
    obs = _toy()
    rng = np.random.default_rng(9)
    pred = rng.random(obs.shape)
    a = evaluate(pred, obs).to_dict()
    b = evaluate(pred, obs).to_dict()
    assert a == b


def test_evaluate_breaks_out_every_lead_time():
    obs = _toy(lead=5)
    r = evaluate(obs, obs, lead_minutes=[30, 60, 120, 240, 360])
    assert [l.lead_minutes for l in r.per_lead] == [30, 60, 120, 240, 360]


def test_evaluate_rejects_bad_input():
    obs = _toy()
    with pytest.raises(ValueError, match="shape mismatch"):
        evaluate(obs[:, :2], obs)
    with pytest.raises(ValueError, match="probabilities"):
        evaluate(obs * 3, obs)
    with pytest.raises(ValueError, match=r"\(N, L, H, W\)"):
        evaluate(obs[0], obs[0])
    with pytest.raises(ValueError, match="lead_minutes"):
        evaluate(obs, obs, lead_minutes=[1, 2])


def test_continuous_obs_are_binarised_by_config():
    """QPE in mm/hr -> events, at the IMD-style cloudburst threshold."""
    rain = np.zeros((1, 1, 4, 4)); rain[0, 0, 1, 1] = 120.0; rain[0, 0, 2, 2] = 40.0
    pred = np.ones((1, 1, 4, 4)) * 0.9
    r = evaluate(pred, rain, EvalConfig(obs_threshold=100.0))
    assert r.pooled["headline"]["hits"] == 1
    assert r.pooled["headline"]["false_alarms"] == 15


def test_drop_masked_false_counts_masked_as_no_event():
    """The non-default path, verified so its cost is visible rather than assumed."""
    pred = np.full((1, 1, 2, 2), 0.9)
    obs = np.zeros((1, 1, 2, 2)); obs[0, 0, 0, 0] = 1
    mask = np.array([[[[True, False], [False, False]]]])

    kept = evaluate(pred, obs, EvalConfig(drop_masked=True), mask=mask)
    assert kept.pooled["headline"]["n"] == 1
    assert kept.pooled["headline"]["false_alarms"] == 0

    folded = evaluate(pred, obs, EvalConfig(drop_masked=False), mask=mask)
    assert folded.pooled["headline"]["n"] == 4
    assert folded.pooled["headline"]["false_alarms"] == 3


def test_summary_table_renders():
    obs = _toy()
    rng = np.random.default_rng(11)
    r = evaluate(rng.random(obs.shape), obs, lead_minutes=[30, 60, 90, 120])
    text = r.summary_table()
    assert "base rate" in text and "CSI" in text and "FSS@50km" in text


# --------------------------------------------------------------------------
# Baselines
# --------------------------------------------------------------------------

def test_persistence_repeats_the_last_frame():
    last = np.arange(2 * 4 * 4, dtype=float).reshape(2, 4, 4) / 100
    p = persistence(last, n_lead=3)
    assert p.shape == (2, 3, 4, 4)
    for li in range(3):
        assert np.array_equal(p[:, li], last)


def test_persistence_is_perfect_on_a_frozen_sky():
    """Sanity anchor: if nothing moves, persistence must score 1.0.
    Any model that cannot beat this at 30 min is broken."""
    rng = np.random.default_rng(12)
    frame = (rng.random((3, 16, 16)) < 0.1).astype(float)
    obs = np.repeat(frame[:, None], 4, axis=1)
    r = evaluate(persistence(frame, 4), obs)
    assert r.pooled["headline"]["csi"] == 1.0
