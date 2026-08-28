"""Tests for the multi-head scorecard.

The behaviour that matters here is not arithmetic -- it is that a head
going backwards cannot hide behind the other two.
"""
import numpy as np
import pytest

from nowcast_eval import EvalConfig, evaluate_multi
from nowcast_eval.multihazard import MultiHazardResult

LEAD = [30, 60, 120, 240]


def _hazard(rate, skill, n=20, h=16, w=16, seed=0):
    rng = np.random.default_rng(seed)
    obs = (rng.random((n, 4, h, w)) < rate).astype(float)
    pred = np.where(obs > 0,
                    rng.uniform(1 - skill, 1.0, obs.shape),
                    rng.uniform(0.0, skill, obs.shape))
    return pred, obs


def _three_heads(skills=(0.9, 0.6, 0.7), seed=0):
    preds, obss = {}, {}
    rates = {"thunderstorm": 2e-2, "cloudburst": 2e-3, "flash_flood": 5e-3}
    for (name, rate), sk in zip(rates.items(), skills):
        p, o = _hazard(rate, sk, seed=seed + hash(name) % 100)
        preds[name], obss[name] = p, o
    return preds, obss


def test_scores_every_head():
    preds, obss = _three_heads()
    r = evaluate_multi(preds, obss, lead_minutes=LEAD)
    assert set(r.names) == {"thunderstorm", "cloudburst", "flash_flood"}
    for name in r.names:
        assert len(r[name].per_lead) == 4


def test_each_head_keeps_its_own_config():
    """Cloudburst and thunderstorm need different thresholds and different
    label definitions; a shared config would be the wrong call."""
    preds, obss = _three_heads()
    cfgs = {
        "thunderstorm": EvalConfig(name="ts", headline_threshold=0.4),
        "cloudburst": EvalConfig(name="cb", headline_threshold=0.7),
        "flash_flood": EvalConfig(name="ff", headline_threshold=0.5),
    }
    r = evaluate_multi(preds, obss, cfgs, lead_minutes=LEAD)
    assert r["thunderstorm"].config["headline_threshold"] == 0.4
    assert r["cloudburst"].config["headline_threshold"] == 0.7


def test_worst_head_is_the_minimum_not_the_mean():
    """The design decision: averaging heads with base rates two orders of
    magnitude apart produces a meaningless number that hides regressions."""
    preds, obss = _three_heads(skills=(0.95, 0.35, 0.8))
    r = evaluate_multi(preds, obss, lead_minutes=LEAD)
    name, val = r.worst_head("csi")
    all_csi = r.headline("csi")
    assert val == pytest.approx(min(all_csi.values()))
    assert val < np.mean(list(all_csi.values())), "must not be the average"


def test_compare_flags_a_regressed_head():
    """The MTL failure mode: one head improves, another silently degrades."""
    preds_a, obss = _three_heads(skills=(0.7, 0.9, 0.7), seed=1)
    r_before = evaluate_multi(preds_a, obss, lead_minutes=LEAD)

    # thunderstorm gets better, cloudburst gets much worse
    preds_b, _ = _three_heads(skills=(0.95, 0.3, 0.7), seed=1)
    preds_b["flash_flood"] = preds_a["flash_flood"]
    r_after = evaluate_multi(preds_b, obss, lead_minutes=LEAD)

    report = r_after.compare(r_before)
    assert "REGRESSED" in report
    assert "cloudburst" in report
    assert "loss weights" in report


def test_compare_is_quiet_when_nothing_regressed():
    preds, obss = _three_heads(seed=2)
    r = evaluate_multi(preds, obss, lead_minutes=LEAD)
    assert "no head regressed" in r.compare(r)


def test_summary_table_shows_every_head_and_the_worst():
    preds, obss = _three_heads()
    text = evaluate_multi(preds, obss, lead_minutes=LEAD).summary_table()
    for name in ("thunderstorm", "cloudburst", "flash_flood"):
        assert name in text
    assert "worst head" in text
    assert "not on the mean" in text


def test_thin_event_counts_are_flagged():
    """A head with too few events to rank must say so in the scorecard."""
    preds, obss = _three_heads()
    p, o = _hazard(1e-5, 0.9, seed=99)      # essentially no events
    preds["cloudburst"], obss["cloudburst"] = p, o
    text = evaluate_multi(preds, obss, lead_minutes=LEAD).summary_table()
    assert "too few events to trust" in text and "cloudburst" in text


def test_mismatched_hazard_keys_are_rejected():
    preds, obss = _three_heads()
    del obss["cloudburst"]
    with pytest.raises(ValueError, match="hazard keys differ"):
        evaluate_multi(preds, obss)


def test_empty_input_is_rejected():
    with pytest.raises(ValueError, match="no hazards"):
        evaluate_multi({}, {})


def test_round_trips_to_json(tmp_path):
    preds, obss = _three_heads()
    r = evaluate_multi(preds, obss, lead_minutes=LEAD)
    path = tmp_path / "multi.json"
    r.to_json(str(path))
    import json
    loaded = json.loads(path.read_text())
    assert set(loaded) == set(r.names)
