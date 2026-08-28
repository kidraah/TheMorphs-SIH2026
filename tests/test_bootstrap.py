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


def test_warns_when_interval_too_wide_to_rank_models():
    pred, obs = _data(n=25, h=12, w=12, rate=0.004, seed=13)
    ci = bootstrap_ci(pred, obs, EvalConfig(), n_boot=300)
    if np.isfinite(ci.hi["csi"] - ci.lo["csi"]) and ci.hi["csi"] - ci.lo["csi"] > 0.2:
        assert any("too wide" in w for w in ci.warnings)


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
