"""The API contract. Buildable before the model exists, which is the point."""
import base64
import json

import numpy as np
import pytest

from nowcast_api import GEOMETRY, HAZARDS, LEAD_MINUTES, build_alerts, stub_predict
from nowcast_api.contract import encode_grid


def test_the_three_hazards_keep_three_geometries():
    """Flattening these into one grid response is the error that scored
    POD 0.80 where the area-weighted truth was 0.0385."""
    d = stub_predict().to_dict()
    got = {f["hazard"]: f["geometry"] for f in d["fields"]}
    assert got == dict(GEOMETRY)
    assert set(HAZARDS) == set(got)


def test_payload_is_small_enough_to_serve():
    """6.2 MB per lead as JSON floats came to 43.7 MB per forecast. Found by
    building the skeleton, which is why it was built before the model."""
    size = len(json.dumps(stub_predict().to_dict()))
    assert size < 2e6, f"{size/1e6:.1f} MB"


def test_grid_round_trips_through_base64():
    g = np.linspace(0, 1, 64 * 64).reshape(64, 64)
    f = encode_grid("thunderstorm", 60, g, downsample=4)
    raw = np.frombuffer(base64.b64decode(f.grid_b64), dtype=np.uint8)
    assert f.grid_shape == (16, 16) and raw.size == 256
    assert raw.max() == 255


def test_downsampling_takes_the_MAX_not_the_mean():
    """Averaging dilutes the isolated high-probability cell the product
    exists to show: a 4 km cloudburst signal averaged over 16 km reads calm."""
    g = np.zeros((16, 16))
    g[3, 3] = 1.0
    f = encode_grid("thunderstorm", 60, g, downsample=4)
    raw = np.frombuffer(base64.b64decode(f.grid_b64), dtype=np.uint8)
    assert raw.max() == 255, "the peak survived"
    assert raw.reshape(4, 4)[0, 0] == 255


def test_every_alert_states_what_its_threshold_costs():
    """An alert that cannot say its FAR is a colour on a map."""
    for a in stub_predict().to_dict()["alerts"]:
        assert 0 <= a["threshold"] <= 1
        assert 0 <= a["far_at_threshold"] <= 1
        assert a["probability"] >= a["threshold"]
        assert a["severity"] in ("advisory", "watch", "warning")


def test_provenance_is_mandatory_and_marks_a_stub_as_a_stub():
    """A stub forecast must never be mistakable for a real one."""
    p = stub_predict().to_dict()["provenance"]
    assert p["model_version"] == "stub"
    assert p["calibrated"] is False
    assert p["registration_gate"] == "NOT_RUN"
    for k in ("cache_fingerprint", "issued_at", "satellite", "scan_time_utc"):
        assert k in p


def test_all_lead_times_arrive_in_one_response():
    """The scrubber is client-side; scrubbing must cost no round trips."""
    d = stub_predict().to_dict()
    assert tuple(d["lead_minutes"]) == LEAD_MINUTES
    for h in HAZARDS:
        leads = {f["lead_minutes"] for f in d["fields"] if f["hazard"] == h}
        assert leads == set(LEAD_MINUTES), h


def test_point_and_basin_fields_carry_their_element_identities():
    """A client must never infer which gauge or basin a value belongs to
    from array order."""
    for f in stub_predict().to_dict()["fields"]:
        if f["geometry"] == "grid":
            continue
        assert len(f["element_ids"]) == len(f["values"])
        assert len(f["element_lonlat"]) == len(f["values"])


def test_alerts_are_ordered_worst_first():
    a = stub_predict().to_dict()["alerts"]
    assert a == sorted(a, key=lambda x: (-x["probability"], x["lead_minutes"]))


def test_thresholds_are_arguments_not_constants():
    d = stub_predict().to_dict()
    few = build_alerts([type("F", (), f)() for f in []], thresholds={"warning": 0.99})
    assert few == []
    from nowcast_api.contract import HazardField
    hf = [HazardField("cloudburst", "point", 60, values=[0.4, 0.9],
                      element_ids=["a", "b"], element_lonlat=[[75, 20], [76, 21]])]
    assert len(build_alerts(hf, thresholds={"advisory": .3, "watch": .5, "warning": .7},
                            far={"advisory": .9, "watch": .8, "warning": .99})) == 2
    assert len(build_alerts(hf, thresholds={"advisory": .95, "watch": .96, "warning": .97},
                            far={"advisory": .9, "watch": .8, "warning": .99})) == 0
