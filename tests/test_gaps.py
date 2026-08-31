"""A known gap is not missing data, and the harness must not confuse them."""
import numpy as np
import pytest

from nowcast_eval.gaps import Gap, GapRegistry, mask_known_gaps

GAP_T = "2015-12-02T11:30:00Z"


def reg():
    return GapRegistry([Gap(identifier="3DIMG_02DEC2015_1130_L1B_STD_V01R00.h5",
                            scan_time_utc=GAP_T,
                            reason="persistent server-side HTTP 500",
                            evidence="500 on two dates", recoverable=False)])


def test_the_shipped_registry_loads_and_declares_the_chennai_gap():
    r = GapRegistry.load()
    assert r.is_known(scan_time_utc=GAP_T)
    assert "Internal Server Error" in r.report() or "500" in r.report()


def test_a_declared_gap_is_masked_not_scored():
    """Scoring an absent slot as 'no event' charges the model with a false
    alarm wherever it correctly forecast rain into the gap."""
    times = np.array([["2015-12-02T11:00:00Z", GAP_T]], dtype=object)
    m, info = mask_known_gaps(np.ones((1, 2, 3, 3), bool), times, reg())
    assert info["slots_masked"] == 1
    assert m[0, 0].all(), "the good slot is untouched"
    assert not m[0, 1].any(), "the gap slot is fully masked"


def test_the_masked_fraction_is_reported():
    """A score without its support is a score over an unstated denominator."""
    times = np.array([["2015-12-02T11:00:00Z", GAP_T]], dtype=object)
    _, info = mask_known_gaps(np.ones((1, 2, 2, 2), bool), times, reg())
    assert info["fraction_masked"] == 0.5
    assert info["gaps_hit"] == [GAP_T]


def test_an_undeclared_absence_is_still_reported_as_a_shortfall():
    """The registry must shrink the unexplained list to the genuinely broken,
    not absorb everything absent."""
    r = reg()
    out = r.unexplained(["3DIMG_02DEC2015_1130_L1B_STD_V01R00.h5", "other.h5"], [])
    assert out == ["other.h5"]


def test_a_recoverable_gap_is_flagged_as_something_to_fix():
    """'We could not get it' is not a reason to carry it forever."""
    r = GapRegistry([Gap(identifier="x.h5", scan_time_utc="t",
                         reason="timeout", recoverable=True)])
    assert "RECOVERABLE" in r.report()


def test_shape_mismatch_is_refused():
    with pytest.raises(ValueError, match="must match"):
        mask_known_gaps(np.ones((2, 3, 4, 4), bool),
                        np.array([["a", "b"]], dtype=object), reg())


def test_masking_never_turns_an_invalid_cell_valid():
    m0 = np.zeros((1, 1, 2, 2), bool)
    m, _ = mask_known_gaps(m0, np.array([[GAP_T]], dtype=object), reg())
    assert not m.any()
