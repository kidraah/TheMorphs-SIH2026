"""Config estimation. Instance 11: a valid config that would pull 69.70 TB.

The estimator is validated against the one real observation there is -- the
config that was aborted at the download prompt returned **179,134 granules,
69.70 TB**. If the model drifts, the first test here fails.
"""
from datetime import date

import pytest

from nowcast_data.mosdac_config import (ConfigTooBroad, Estimate, audit,
                                        estimate, validate, write)

TODAY = date(2026, 8, 30)


def cfg(**sp):
    base = {"datasetId": "3RIMG_L1B_STD", "startTime": "", "endTime": "",
            "count": "", "boundingBox": "62.3,5.8,101.7,39.5", "gId": ""}
    base.update(sp)
    return {"search_parameters": base}


def test_reproduces_the_observed_catastrophe():
    """THE known answer. Empty start/end matched the whole 3RIMG archive."""
    e = estimate(cfg(), today=TODAY)
    assert e.unbounded
    assert abs(e.granules - 179_134) / 179_134 < 0.10, e.report()
    assert abs(e.gb / 1024 - 69.70) / 69.70 < 0.15, e.report()


def test_empty_dates_are_refused_with_the_reason():
    with pytest.raises(ConfigTooBroad, match="unbounded range, not an unset field"):
        validate(cfg(), today=TODAY)


def test_a_key_the_client_never_reads_is_named_in_the_error():
    """The dates were in `_one_job_per_date`, on the assumption the client
    would iterate. It takes one range per file."""
    c = cfg()
    c["_one_job_per_date"] = [{"startTime": "2019-07-25"}]
    with pytest.raises(ConfigTooBroad, match=r"_one_job_per_date"):
        validate(c, today=TODAY)


def test_only_one_empty_endpoint_is_still_unbounded():
    for sp in ({"startTime": "2019-07-25"}, {"endTime": "2019-07-25"}):
        assert estimate(cfg(**sp), today=TODAY).unbounded


def test_a_single_day_is_one_day_of_scans():
    e = estimate(cfg(startTime="2019-07-25", endTime="2019-07-25"), today=TODAY)
    assert e.days == 1 and e.granules == 48
    validate(cfg(startTime="2019-07-25", endTime="2019-07-25"), today=TODAY)


def test_a_fifteen_day_window_is_refused_at_the_default_limit():
    """The archive configs are exactly this: 720 granules, 315 GB each."""
    c = cfg(startTime="2023-06-01", endTime="2023-06-15")
    e = estimate(c, today=TODAY)
    assert e.granules == 720 and 300 < e.gb < 330
    with pytest.raises(ConfigTooBroad, match="above the 200"):
        validate(c, today=TODAY)
    validate(c, max_granules=1000, today=TODAY)      # deliberate bulk is allowed


def test_count_caps_the_estimate():
    c = cfg(startTime="2023-06-01", endTime="2023-06-15", count=10)
    assert estimate(c, today=TODAY).granules == 10


def test_write_refuses_before_touching_disk(tmp_path):
    p = tmp_path / "bad.json"
    with pytest.raises(ConfigTooBroad):
        write(p, cfg(), today=TODAY)
    assert not p.exists(), "a too-broad config must never reach disk"


def test_write_creates_a_good_config(tmp_path):
    p = tmp_path / "good.json"
    e = write(p, cfg(startTime="2019-07-25", endTime="2019-07-25"), today=TODAY)
    assert p.exists() and e.granules == 48


def test_unknown_dataset_falls_back_rather_than_crashing():
    e = estimate(cfg(datasetId="SOMETHING_NEW", startTime="2024-01-01",
                     endTime="2024-01-01"), today=TODAY)
    assert e.granules > 0


def test_unparseable_date_raises_rather_than_being_treated_as_empty():
    """Silently treating a bad date as unbounded is how this happened."""
    with pytest.raises(ValueError, match="unparseable"):
        estimate(cfg(startTime="25-07-2019", endTime="2019-07-25"), today=TODAY)


# ---------------------------------------------------------------------------
# The repo's own configs
# ---------------------------------------------------------------------------

def test_no_shipped_config_has_an_unbounded_range():
    """The failure that cost a 69.70 TB near-miss must not recur in-tree."""
    rows = audit("configs/mosdac", today=TODAY)
    unbounded = [r["file"] for r in rows if r.get("unbounded")]
    assert not unbounded, f"unbounded date range in: {unbounded}"


def test_every_gate_and_event_config_is_within_the_limit():
    rows = audit("configs/mosdac", max_granules=200, today=TODAY)
    over = [r["file"] for r in rows
            if r.get("over_limit") and "_archive_" not in r["file"]]
    assert not over, f"over 200 granules: {over}"


def test_the_archive_configs_are_flagged_and_their_true_size_is_recorded():
    """They are NOT the empty-date bug -- their dates are correct. They are a
    different mismatch: 15-day CONTINUOUS windows, where the plan is
    event-sampled. Recorded as a number so it cannot drift quietly."""
    # "_archive_", not "archive": 02_verify_3D_archive.json is a one-granule
    # format check and matched the looser substring.
    rows = [r for r in audit("configs/mosdac", today=TODAY)
            if "_archive_" in r["file"] and not r.get("error")]
    assert len(rows) == 24
    assert all(r["over_limit"] for r in rows)
    total_tb = sum(r["gb"] for r in rows) / 1024
    assert 7.0 < total_tb < 8.0, f"{total_tb:.1f} TB"
