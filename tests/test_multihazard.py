"""Tests for the multi-head scorecard.

Fixture note
------------
An earlier version of this file had two defects worth recording, because
both passed CI while testing nothing:

  1. it seeded from `hash(name)`, which Python randomises per process, so
     the data changed on every run;
  2. its `skill` parameter was inverted -- `skill=0.35` produced a PERFECT
     forecast -- so the regression test asserted "REGRESSED" while actually
     moving the head from bad to perfect.

Hence `_hazard` below controls POD and the false alarm rate directly, and
`test_fixture_is_honest` fails if the quality knob ever stops meaning what
its name says.
"""
import numpy as np
import pytest

from nowcast_eval import (EvalConfig, attach_confidence_intervals, contingency,
                          evaluate_multi)

LEAD = [30, 60, 120, 240]
RATES = {"thunderstorm": 2e-2, "cloudburst": 2e-3, "flash_flood": 5e-3}
SEEDS = {"thunderstorm": 11, "cloudburst": 22, "flash_flood": 33}   # fixed, not hash()


def _hazard(rate, quality, n=40, h=32, w=32, seed=0):
    """quality in [0, 1]: 0 = no skill (POD == false alarm rate), 1 = near perfect.

    POD and the false alarm RATE are set directly so the knob cannot invert.
    """
    rng = np.random.default_rng(seed)
    obs = (rng.random((n, 4, h, w)) < rate).astype(float)
    pod = 0.50 + 0.49 * quality
    f = 0.50 - 0.49 * quality
    fires = np.where(obs > 0, rng.random(obs.shape) < pod, rng.random(obs.shape) < f)
    pred = np.where(fires, 0.90, 0.10) + rng.uniform(-0.05, 0.05, obs.shape)
    return np.clip(pred, 0, 1), obs


def _three_heads(qualities=(0.9, 0.6, 0.7)):
    preds, obss = {}, {}
    for (name, rate), q in zip(RATES.items(), qualities):
        preds[name], obss[name] = _hazard(rate, q, seed=SEEDS[name])
    return preds, obss


def _with_cis(preds, obss, cfgs=None, n_boot=200):
    r = evaluate_multi(preds, obss, cfgs, lead_minutes=LEAD)
    for name in r.names:
        cfg = cfgs[name] if cfgs else EvalConfig(name=name)
        attach_confidence_intervals(r[name], preds[name], obss[name], cfg,
                                    n_boot=n_boot)
    return r


# --------------------------------------------------------------------------
# The fixture must be capable of failing
# --------------------------------------------------------------------------

def test_fixture_is_honest():
    """Higher quality must mean better, and mid quality must be imperfect.

    Without this the suite can pass on data where failure is impossible.
    """
    obs_rate = 2e-2
    scores = []
    for q in (0.2, 0.5, 0.8, 0.95):
        pred, obs = _hazard(obs_rate, q, seed=7)
        t = contingency(pred, obs, 0.5)
        scores.append(t.sedi)
        if q < 0.9:
            assert t.hits > 0 and t.misses > 0 and t.false_alarms > 0, (
                f"quality={q} must produce real misses AND false alarms")
    assert all(b > a for a, b in zip(scores, scores[1:])), (
        f"SEDI must increase with quality, got {scores}")


def test_fixture_is_deterministic_across_processes():
    """No hash(), no time, no global RNG -- same bytes every run."""
    a, _ = _hazard(2e-2, 0.7, seed=5)
    b, _ = _hazard(2e-2, 0.7, seed=5)
    assert np.array_equal(a, b)


# --------------------------------------------------------------------------
# Basic wiring
# --------------------------------------------------------------------------

def test_scores_every_head():
    preds, obss = _three_heads()
    r = evaluate_multi(preds, obss, lead_minutes=LEAD)
    assert set(r.names) == set(RATES)
    assert all(len(r[n].per_lead) == 4 for n in r.names)


def test_each_head_keeps_its_own_config():
    preds, obss = _three_heads()
    cfgs = {n: EvalConfig(name=n, headline_threshold=t)
            for n, t in zip(RATES, (0.4, 0.7, 0.5))}
    r = evaluate_multi(preds, obss, cfgs, lead_minutes=LEAD)
    assert r["thunderstorm"].config["headline_threshold"] == 0.4
    assert r["cloudburst"].config["headline_threshold"] == 0.7


# --------------------------------------------------------------------------
# Metric choice: why the minimum is taken over SEDI, not CSI
# --------------------------------------------------------------------------

def test_csi_ranks_heads_by_rarity_even_at_identical_quality():
    """The reason worst_head defaults to SEDI.

    Give all three heads the SAME forecast quality. CSI still spreads them
    by nearly an order of magnitude and ranks the rarest one worst -- that
    spread is a base-rate artefact, not a quality difference, so a CSI
    minimum across heads would select the rarest hazard by construction.
    SEDI, being base-rate independent, keeps them together.

    (CSI does still respond to quality -- it conflates the two. That is
    precisely why it is not comparable across heads with different base
    rates, and why it stays a reporting metric here rather than a
    selection one.)
    """
    preds, obss = {}, {}
    for name, rate in RATES.items():
        preds[name], obss[name] = _hazard(rate, 0.8, seed=SEEDS[name])
    r = evaluate_multi(preds, obss, lead_minutes=LEAD)

    csi = r.headline("csi")
    sedi = r.headline("sedi")

    assert max(csi.values()) / min(csi.values()) > 5, (
        f"CSI must spread widely on identical quality, got {csi}")
    assert min(csi, key=csi.get) == min(RATES, key=RATES.get), (
        "CSI's worst head must be the rarest one, showing the artefact")
    assert max(sedi.values()) - min(sedi.values()) < 0.05, (
        f"SEDI must stay flat across base rates, got {sedi}")


def test_worst_head_is_the_minimum_not_the_mean():
    preds, obss = _three_heads(qualities=(0.95, 0.35, 0.8))
    r = evaluate_multi(preds, obss, lead_minutes=LEAD)
    name, val = r.worst_head("sedi")
    all_sedi = r.headline("sedi")
    assert val == pytest.approx(min(all_sedi.values()))
    assert val < np.mean(list(all_sedi.values()))
    assert name == "cloudburst"


# --------------------------------------------------------------------------
# Checkpointing on the lower bound
# --------------------------------------------------------------------------

def test_lower_bound_requires_confidence_intervals():
    """Fails loudly rather than silently reverting to the point estimate."""
    preds, obss = _three_heads()
    r = evaluate_multi(preds, obss, lead_minutes=LEAD)
    with pytest.raises(ValueError, match="no confidence intervals"):
        r.worst_head_lower_bound("sedi")


def test_lower_bound_is_stricter_than_the_point_estimate():
    preds, obss = _three_heads()
    r = _with_cis(preds, obss)
    _, lo = r.worst_head_lower_bound("sedi")
    _, pt = r.worst_head("sedi")
    assert lo <= pt, "the defensible number cannot exceed the lucky one"


def test_an_unmeasurable_head_blocks_checkpointing_instead_of_being_dropped():
    """The bug this test caught: a head with too few events to define SEDI
    was silently excluded from selection, so checkpointing reported a
    healthy score while blind to it -- and the blind head is always the
    rarest, most important hazard, because that is where evidence runs out
    first. It must rank worst instead.
    """
    solid, obs_solid = _hazard(2e-2, 0.80, n=60, seed=1)
    thin, obs_thin = _hazard(2e-4, 0.85, n=8, h=16, w=16, seed=2)

    r = evaluate_multi({"solid": solid, "thin": thin},
                       {"solid": obs_solid, "thin": obs_thin},
                       lead_minutes=LEAD)
    for k, (p, o) in {"solid": (solid, obs_solid), "thin": (thin, obs_thin)}.items():
        attach_confidence_intervals(r[k], p, o, EvalConfig(name=k), n_boot=300)

    assert not np.isfinite(r["thin"].pooled["ci"]["lo"]["sedi"]), (
        "fixture must produce an unmeasurable head or this proves nothing")
    assert np.isfinite(r["solid"].pooled["ci"]["lo"]["sedi"])

    name, val = r.worst_head_lower_bound("sedi")
    assert name == "thin", "the unmeasurable head must be selected as worst"
    assert val == float("-inf")
    assert "CHECKPOINT BLOCKED" in r.summary_table()


def test_lower_bound_ranks_the_weaker_head_when_both_are_measurable():
    preds, obss = _three_heads(qualities=(0.9, 0.45, 0.8))
    r = _with_cis(preds, obss, n_boot=300)
    name, val = r.worst_head_lower_bound("sedi")
    assert name == "cloudburst"
    assert np.isfinite(val)


# --------------------------------------------------------------------------
# The MTL failure mode
# --------------------------------------------------------------------------

def test_compare_flags_a_regressed_head():
    """thunderstorm improves, cloudburst genuinely gets worse."""
    before_p, obss = _three_heads(qualities=(0.6, 0.9, 0.7))
    after_p, _ = _three_heads(qualities=(0.9, 0.35, 0.7))

    r_before = evaluate_multi(before_p, obss, lead_minutes=LEAD)
    r_after = evaluate_multi(after_p, obss, lead_minutes=LEAD)

    # direction check first, so the assertion below cannot pass vacuously
    assert r_after.headline("sedi")["thunderstorm"] > r_before.headline("sedi")["thunderstorm"]
    assert r_after.headline("sedi")["cloudburst"] < r_before.headline("sedi")["cloudburst"]

    report = r_after.compare(r_before, metric="sedi")
    assert "REGRESSED" in report and "cloudburst" in report
    assert "loss weights" in report


def test_compare_is_quiet_when_nothing_regressed():
    preds, obss = _three_heads()
    r = evaluate_multi(preds, obss, lead_minutes=LEAD)
    assert "no head regressed" in r.compare(r, metric="sedi")


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

def test_summary_table_names_the_checkpoint_metric():
    preds, obss = _three_heads()
    text = _with_cis(preds, obss).summary_table()
    for name in RATES:
        assert name in text
    assert "CHECKPOINT ON" in text and "lower bound" in text
    assert "not comparable across" in text


def test_summary_table_without_cis_says_so():
    preds, obss = _three_heads()
    text = evaluate_multi(preds, obss, lead_minutes=LEAD).summary_table()
    assert "point estimate only" in text


def test_thin_event_counts_are_flagged():
    preds, obss = _three_heads()
    preds["cloudburst"], obss["cloudburst"] = _hazard(1e-5, 0.9, seed=99)
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
    assert set(json.loads(path.read_text())) == set(r.names)
