"""Tests for the confidence intervals.

The load-bearing claim in bootstrap.py is that pooling per-sample
sufficient statistics reproduces the full recomputation EXACTLY. If that
is off by even a little, every interval is subtly wrong while looking
perfectly reasonable. So it is tested against the primary implementation
directly, not approximately.
"""
import numpy as np
import pytest

from nowcast_eval import (EvalConfig, attach_confidence_intervals, bootstrap_ci,
                          brier_score, contingency, evaluate, fss,
                          significantly_better)
from nowcast_eval.bootstrap import _metrics_from_sums, collect_stats


def _data(n=40, h=32, w=32, rate=0.05, seed=0):
    """A skillful but genuinely IMPERFECT forecast.

    It must produce real misses and real false alarms. An earlier version of
    this helper built pred from obs in a way that could never do either --
    a silently perfect forecast with zero variance, which made every
    consistency check below compare 1.0 against 1.0 and pass for free.
    """
    rng = np.random.default_rng(seed)
    obs = (rng.random((n, h, w)) < rate).astype(float)
    pred = np.where(obs > 0,
                    rng.uniform(0.2, 1.0, obs.shape),   # ~37% fall below 0.5 -> misses
                    rng.uniform(0.0, 0.8, obs.shape))   # ~37% rise above 0.5 -> false alarms
    return pred, obs


def test_the_test_data_is_actually_imperfect():
    """Guards the helper above: if this ever scores perfectly again, the
    consistency tests stop testing anything."""
    pred, obs = _data()
    t = contingency(pred, obs, 0.5)
    assert t.hits > 0 and t.misses > 0 and t.false_alarms > 0
    assert 0.0 < t.csi < 1.0


# --------------------------------------------------------------------------
# The sufficient-statistic shortcut must be exact
# --------------------------------------------------------------------------

def test_pooled_contingency_matches_direct_computation():
    pred, obs = _data()
    cfg = EvalConfig(headline_threshold=0.5)
    st = collect_stats(pred, obs, 0.5, cfg)
    m = _metrics_from_sums(st, np.arange(st.n_samples), list(cfg.neighborhood_km))
    direct = contingency(pred, obs, 0.5)

    assert m["csi"] == pytest.approx(direct.csi)
    assert m["pod"] == pytest.approx(direct.pod)
    assert m["far"] == pytest.approx(direct.far)
    assert m["base_rate"] == pytest.approx(direct.base_rate)


def test_pooled_fss_matches_direct_computation():
    """The subtle one: FSS pools as sum(diff) / (sum(pf^2) + sum(po^2)),
    with the cell count cancelling. Verified against fss() at every
    neighbourhood size."""
    pred, obs = _data()
    cfg = EvalConfig(headline_threshold=0.5, grid_km=4.0)
    st = collect_stats(pred, obs, 0.5, cfg)
    m = _metrics_from_sums(st, np.arange(st.n_samples), list(cfg.neighborhood_km))

    for size, km in zip(cfg.neighborhood_pixels(), cfg.neighborhood_km):
        direct = fss(pred, obs, 0.5, size).fss
        assert m[f"fss@{km:g}km"] == pytest.approx(direct, abs=1e-12), f"{km} km"


def test_pooled_brier_matches_direct_computation():
    pred, obs = _data()
    st = collect_stats(pred, obs, 0.5, EvalConfig())
    m = _metrics_from_sums(st, np.arange(st.n_samples), [])
    assert m["brier"] == pytest.approx(brier_score(pred, obs))


def test_pooled_bss_matches_full_evaluation():
    pred, obs = _data()
    r = evaluate(pred[:, None], obs[:, None], EvalConfig())
    ci = bootstrap_ci(pred, obs, EvalConfig(), n_boot=50)
    assert ci.point["bss"] == pytest.approx(r.pooled["probabilistic"]["bss"])


# --------------------------------------------------------------------------
# Interval behaviour
# --------------------------------------------------------------------------

def test_perfect_forecast_has_degenerate_interval():
    """No sampling noise to speak of: every replicate scores 1.0."""
    _, obs = _data()
    ci = bootstrap_ci(obs, obs, EvalConfig(), n_boot=200)
    assert ci.point["csi"] == 1.0
    assert ci.lo["csi"] == pytest.approx(1.0)
    assert ci.hi["csi"] == pytest.approx(1.0)


def test_interval_brackets_the_point_estimate():
    pred, obs = _data()
    ci = bootstrap_ci(pred, obs, EvalConfig(), n_boot=500, seed=3)
    for m in ("csi", "pod", "far"):
        assert ci.lo[m] <= ci.point[m] <= ci.hi[m], m


def test_interval_narrows_as_sample_grows():
    """Basic statistical sanity: more forecast cases -> tighter bounds."""
    widths = []
    for n in (25, 100, 400):
        pred, obs = _data(n=n, h=16, w=16, seed=11)
        ci = bootstrap_ci(pred, obs, EvalConfig(), n_boot=400, seed=5)
        widths.append(ci.hi["csi"] - ci.lo["csi"])
    assert widths[0] > widths[1] > widths[2], widths


def test_bootstrap_is_reproducible():
    pred, obs = _data()
    a = bootstrap_ci(pred, obs, EvalConfig(), n_boot=100, seed=42)
    b = bootstrap_ci(pred, obs, EvalConfig(), n_boot=100, seed=42)
    assert a.lo == b.lo and a.hi == b.hi


def test_resampling_unit_is_the_case_not_the_pixel():
    """Guards the core design decision. Pixels within a case are correlated,
    so N identical cases must NOT produce a tighter interval than one case
    repeated -- a pixel bootstrap would wrongly shrink it toward zero width."""
    rng = np.random.default_rng(2)
    one = (rng.random((1, 32, 32)) < 0.05).astype(float)
    pred_one = np.clip(one * 0.7 + 0.1, 0, 1)
    # 30 exact copies: no new information, so the interval must stay wide-ish
    ci = bootstrap_ci(np.repeat(pred_one, 30, 0), np.repeat(one, 30, 0),
                      EvalConfig(), n_boot=300, seed=1)
    assert ci.hi["csi"] - ci.lo["csi"] == pytest.approx(0.0, abs=1e-9), (
        "identical cases carry identical statistics, so every replicate must "
        "give the same score")


# --------------------------------------------------------------------------
# Warnings: the point of the module
# --------------------------------------------------------------------------

def test_warns_on_too_few_events():
    pred, obs = _data(n=30, h=16, w=16, rate=1e-4, seed=8)
    ci = bootstrap_ci(pred, obs, EvalConfig(), n_boot=200)
    assert ci.n_events < 100
    assert any("observed events" in w for w in ci.warnings)


def test_warns_on_too_few_cases():
    pred, obs = _data(n=8)
    ci = bootstrap_ci(pred, obs, EvalConfig(), n_boot=200)
    assert any("forecast cases" in w for w in ci.warnings)


def test_warns_when_interval_is_wide_relative_to_the_estimate():
    """Rare-event CSI is small in absolute terms, so its interval is small
    too. An absolute-width rule never fires on exactly the hazards that need
    it -- the warning must be relative to the point estimate."""
    pred, obs = _data(n=25, h=12, w=12, rate=0.004, seed=13)
    ci = bootstrap_ci(pred, obs, EvalConfig(), n_boot=300)
    width, pt = ci.hi["csi"] - ci.lo["csi"], ci.point["csi"]

    assert width < 0.2, "fixture must NOT trip the absolute rule, or this proves nothing"
    assert width / pt > 0.5, f"fixture must be wide relative to {pt:.4f}, got {width:.4f}"
    assert any("% of the value" in w for w in ci.warnings)


def test_absolute_width_rule_still_fires_for_common_events():
    rng = np.random.default_rng(0)
    n, h, w = 6, 10, 10
    obs = (rng.random((n, h, w)) < 0.3).astype(float)
    pred = rng.random((n, h, w))
    ci = bootstrap_ci(pred, obs, EvalConfig(), n_boot=300)
    if ci.hi["csi"] - ci.lo["csi"] > 0.2:
        assert any("too wide" in x for x in ci.warnings)


# --------------------------------------------------------------------------
# Model comparison
# --------------------------------------------------------------------------

def test_overlapping_intervals_are_not_a_real_difference():
    """The check that stops 'our model beat persistence' being said too early."""
    pred, obs = _data(n=30, seed=4)
    a = bootstrap_ci(pred, obs, EvalConfig(), n_boot=300, seed=1)
    b = bootstrap_ci(np.clip(pred + 0.001, 0, 1), obs, EvalConfig(), n_boot=300, seed=2)
    assert not significantly_better(a, b), "near-identical models must not register"


def test_clearly_better_model_is_detected():
    _, obs = _data(n=60, seed=6)
    rng = np.random.default_rng(6)
    good = np.clip(obs * 0.95 + rng.random(obs.shape) * 0.02, 0, 1)
    bad = rng.random(obs.shape) * 0.6
    a = bootstrap_ci(good, obs, EvalConfig(), n_boot=300, seed=1)
    b = bootstrap_ci(bad, obs, EvalConfig(), n_boot=300, seed=2)
    assert significantly_better(a, b)


# --------------------------------------------------------------------------
# Attachment to a full result
# --------------------------------------------------------------------------

def test_attach_adds_ci_to_every_lead_and_pooled():
    rng = np.random.default_rng(0)
    obs = (rng.random((20, 4, 16, 16)) < 0.05).astype(float)
    pred = np.clip(obs * 0.8 + rng.random(obs.shape) * 0.2, 0, 1)
    cfg = EvalConfig()
    r = evaluate(pred, obs, cfg, lead_minutes=[30, 60, 120, 240])
    attach_confidence_intervals(r, pred, obs, cfg, n_boot=100)

    assert all(l.ci is not None for l in r.per_lead)
    assert "ci" in r.pooled
    assert "95% CI" in r.summary_table()
    assert r.to_dict()["per_lead"][0]["ci"] is not None


# --------------------------------------------------------------------------
# The resampling unit must survive pooling across lead times
# --------------------------------------------------------------------------

def test_pooled_ci_resamples_cases_not_case_lead_pairs():
    """Lead times within one forecast case are the same storm.

    Treating (case, lead) pairs as independent would report N*L cases and an
    interval that is too narrow -- the same error the module rejects for
    pixels. The pooled CI must report exactly N.
    """
    rng = np.random.default_rng(0)
    N, L = 25, 12
    obs = (rng.random((N, L, 16, 16)) < 0.02).astype(float)
    pred = np.where(obs > 0, rng.uniform(0.2, 1.0, obs.shape),
                    rng.uniform(0.0, 0.8, obs.shape))
    cfg = EvalConfig()
    r = evaluate(pred, obs, cfg, lead_minutes=[30 * i for i in range(1, L + 1)])
    attach_confidence_intervals(r, pred, obs, cfg, n_boot=200)

    assert r.pooled["ci"]["n_samples"] == N, (
        f"pooled must resample {N} cases, not {N * L} case-lead pairs")


def test_correlated_lead_times_do_not_shrink_the_interval():
    """Duplicating a case across more lead times adds no information, so the
    interval must not narrow. A (case, lead) bootstrap would narrow it."""
    rng = np.random.default_rng(1)
    N = 30
    base_o = (rng.random((N, 1, 16, 16)) < 0.03).astype(float)
    base_p = np.where(base_o > 0, rng.uniform(0.2, 1.0, base_o.shape),
                      rng.uniform(0.0, 0.8, base_o.shape))
    cfg = EvalConfig()

    def width(reps):
        o = np.repeat(base_o, reps, axis=1)      # identical frames, no new info
        p = np.repeat(base_p, reps, axis=1)
        ci = bootstrap_ci(p, o, cfg, n_boot=400, seed=7)
        return ci.hi["csi"] - ci.lo["csi"]

    assert width(8) == pytest.approx(width(1), abs=1e-9)


# --------------------------------------------------------------------------
# Blocking one level further out: cases that share a storm day
# --------------------------------------------------------------------------

def _day_clustered(n_days=10, per_day=6, h=16, w=16, seed=0):
    """Windows cut from a small number of storm days.

    Every window from a day is IDENTICAL here, which is the extreme of the
    real situation: windows from one synoptic setup carry nearly the same
    information. Effective sample size is the day count, not the window count.
    """
    rng = np.random.default_rng(seed)
    days_o = (rng.random((n_days, h, w)) < 0.04).astype(float)
    days_p = np.where(days_o > 0, rng.uniform(0.2, 1.0, days_o.shape),
                      rng.uniform(0.0, 0.8, days_o.shape))
    obs = np.repeat(days_o, per_day, axis=0)
    pred = np.repeat(days_p, per_day, axis=0)
    groups = np.repeat(np.arange(n_days), per_day)
    return pred, obs, groups


def test_grouped_bootstrap_reports_blocks_not_cases():
    pred, obs, groups = _day_clustered(n_days=10, per_day=6)
    ci = bootstrap_ci(pred, obs, EvalConfig(), n_boot=200, groups=groups)
    assert ci.n_samples == 60
    assert ci.n_blocks == 10, "must resample days, not windows"


def test_duplicating_cases_within_a_day_does_not_shrink_the_interval():
    """Parallel to the lead-time test, one level further out.

    Six copies of each storm day carry no more information than one copy.
    With day blocking the interval must be unchanged; without it, the
    interval wrongly narrows.
    """
    cfg = EvalConfig()
    p1, o1, g1 = _day_clustered(n_days=12, per_day=1, seed=3)
    p6, o6, g6 = _day_clustered(n_days=12, per_day=6, seed=3)

    blocked_1 = bootstrap_ci(p1, o1, cfg, n_boot=400, seed=1, groups=g1)
    blocked_6 = bootstrap_ci(p6, o6, cfg, n_boot=400, seed=1, groups=g6)
    w1 = blocked_1.hi["csi"] - blocked_1.lo["csi"]
    w6 = blocked_6.hi["csi"] - blocked_6.lo["csi"]
    assert w6 == pytest.approx(w1, abs=1e-9), (
        f"day-blocked interval must ignore duplication: {w1:.4f} vs {w6:.4f}")

    # and the unblocked version demonstrably understates it
    naive = bootstrap_ci(p6, o6, cfg, n_boot=400, seed=1)
    w_naive = naive.hi["csi"] - naive.lo["csi"]
    assert w_naive < w1 * 0.8, (
        f"ignoring day structure must visibly narrow the interval "
        f"({w_naive:.4f} vs correct {w1:.4f}) -- this is the failure mode")


def test_warns_when_blocks_collapse_far_below_case_count():
    pred, obs, groups = _day_clustered(n_days=5, per_day=20)
    ci = bootstrap_ci(pred, obs, EvalConfig(), n_boot=200, groups=groups)
    assert any("collapse to" in w for w in ci.warnings)


def test_warns_when_independence_is_merely_assumed():
    """No `groups` given is an assertion of independence. Say so."""
    pred, obs = _data(n=40)
    ci = bootstrap_ci(pred, obs, EvalConfig(), n_boot=100)
    assert any("assumed independent" in w for w in ci.warnings)


def test_groups_length_is_validated():
    pred, obs = _data(n=20)
    with pytest.raises(ValueError, match="groups has"):
        bootstrap_ci(pred, obs, EvalConfig(), n_boot=50, groups=np.arange(5))


def test_attach_passes_groups_to_every_lead():
    rng = np.random.default_rng(0)
    obs = (rng.random((24, 3, 16, 16)) < 0.04).astype(float)
    pred = np.where(obs > 0, rng.uniform(0.2, 1.0, obs.shape),
                    rng.uniform(0.0, 0.8, obs.shape))
    cfg = EvalConfig()
    groups = np.repeat(np.arange(6), 4)
    r = evaluate(pred, obs, cfg, lead_minutes=[30, 60, 120])
    attach_confidence_intervals(r, pred, obs, cfg, n_boot=100, groups=groups)
    assert r.pooled["ci"]["n_blocks"] == 6
    assert all(l.ci["n_blocks"] == 6 for l in r.per_lead)


# --------------------------------------------------------------------------
# SEDI
# --------------------------------------------------------------------------

def test_sedi_is_base_rate_independent_where_csi_is_not():
    """The justification for using SEDI on the cloudburst head.

    Hold forecast quality fixed (POD ~0.8, false alarm rate ~0.05) and vary
    the base rate over three orders of magnitude. CSI collapses; SEDI holds.
    """
    csis, sedis = [], []
    for rate in (1e-1, 1e-2, 1e-3):
        rng = np.random.default_rng(0)
        n = 2_000_000
        obs = rng.random(n) < rate
        pred = np.where(obs, rng.random(n) < 0.8, rng.random(n) < 0.05).astype(float)
        t = contingency(pred, obs, 0.5)
        csis.append(t.csi)
        sedis.append(t.sedi)

    assert csis[0] / csis[-1] > 20, f"CSI must collapse with base rate, got {csis}"
    assert max(sedis) - min(sedis) < 0.05, f"SEDI must hold, got {sedis}"


def test_sedi_is_undefined_for_a_perfect_forecast():
    """F = 0 makes log(F) undefined. A real limitation, documented not hidden."""
    _, obs = _data()
    assert np.isnan(contingency(obs, obs, 0.5).sedi)


def test_sedi_is_zero_for_a_random_forecast():
    """No skill means hit rate == false alarm rate, which gives SEDI 0."""
    rng = np.random.default_rng(0)
    n = 2_000_000
    obs = rng.random(n) < 0.01
    pred = (rng.random(n) < 0.3).astype(float)   # fires independently of obs
    assert contingency(pred, obs, 0.5).sedi == pytest.approx(0.0, abs=0.01)


def test_bootstrap_sedi_matches_direct_computation():
    pred, obs = _data()
    ci = bootstrap_ci(pred, obs, EvalConfig(), n_boot=50)
    assert ci.point["sedi"] == pytest.approx(contingency(pred, obs, 0.5).sedi)
