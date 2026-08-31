"""The API contract the dashboard consumes. Model-independent by construction.

Everything here is defined against a `predict()` that returns arrays of the
right shape, so the dashboard and the alerting API can be built, tested and
demonstrated before the model exists -- and so that integration problems
surface now rather than in the last week.

THE THREE HAZARDS HAVE THREE GEOMETRIES
---------------------------------------
This is not a detail the API can paper over. It is the same distinction the
harness enforces (EvalConfig.geometry), and flattening it here would put it
back:

    thunderstorm   GRID   (H, W) probability per 4 km cell
    cloudburst     POINT  one probability per gauge location
    flash_flood    BASIN  one probability per sub-basin polygon

A single "grid of probabilities" response would force the flood head onto the
grid, which is the error that scored POD 0.80 where the truth was 0.0385.

WHAT IS DELIBERATELY NOT IN v1
------------------------------
No user accounts, no historical archive queries, no model management. The
minimum that puts a real forecast on a map with a scrubber, an inspector and
an alert feed -- and nothing that can be added later without breaking a
consumer.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone

API_VERSION = "1.0"
HAZARDS = ("thunderstorm", "cloudburst", "flash_flood")
GEOMETRY = {"thunderstorm": "grid", "cloudburst": "point", "flash_flood": "basin"}

# Lead times the model emits, minutes. 2-6 h is the problem statement's claim.
LEAD_MINUTES = (30, 60, 120, 180, 240, 300, 360)

# Alert bands. These are OPERATING POINTS, not colours: each is a threshold on
# calibrated probability chosen at a stated FAR, and it must be traceable to
# the scorecard that produced it. A band whose threshold nobody can attribute
# is a colour on a map.
SEVERITY = ("advisory", "watch", "warning")


@dataclass
class GridSpec:
    """The analysis grid, sent once so the client need not guess."""
    shape: tuple = (912, 864)
    resolution_km: float = 4.0
    proj: str = "laea"
    lat_0: float = 23.0
    lon_0: float = 82.0
    # Corner lon/lat, for a client that cannot reproject.
    bounds_lonlat: tuple = (62.283, 5.796, 101.717, 39.491)


@dataclass
class Provenance:
    """Attached to EVERY forecast. Non-optional on purpose.

    A risk map with no provenance cannot be audited after the fact, and this
    project's whole discipline is that a number without its context is not a
    number. `registration_gate` is here because a forecast whose registration
    has not passed is displaced on the map while scoring perfectly -- the
    consumer has to be able to see that.
    """
    model_version: str = "stub"
    cache_fingerprint: str = ""
    issued_at: str = ""
    satellite: str = ""
    scan_time_utc: str = ""
    registration_gate: str = "NOT_RUN"     # PASS | FAIL | INCONCLUSIVE | NOT_RUN
    calibrated: bool = False
    notes: str = ""


# A 912x864 grid of JSON floats is 6.2 MB per lead; seven leads plus the
# other two hazards came to 43.7 MB per forecast. Found by building the
# skeleton against a stub, which is the entire reason for building it early.
#
# Grids therefore travel as base64 uint8 at a stated downsample factor. The
# dashboard draws at screen resolution and cannot show 912x864 anyway; exact
# values come from /cell, which returns one location at full precision. 1/4
# gives 228x216 = 49 KB per lead.
GRID_DOWNSAMPLE = 4


@dataclass
class HazardField:
    hazard: str
    geometry: str
    lead_minutes: int
    # point/basin: one value per element. grid: EMPTY -- see `grid_png`.
    values: list = field(default_factory=list)
    # grid only: base64 uint8, row-major, probability * 255.
    grid_b64: str = ""
    grid_shape: tuple = ()
    grid_downsample: int = GRID_DOWNSAMPLE
    # point/basin only: the element identities, so a client never has to
    # infer them from array order.
    element_ids: list = field(default_factory=list)
    element_lonlat: list = field(default_factory=list)


@dataclass
class Alert:
    alert_id: str
    hazard: str
    severity: str
    probability: float
    threshold: float
    far_at_threshold: float          # what this operating point costs
    valid_from: str
    valid_to: str
    lead_minutes: int
    place: str = ""
    element_id: str = ""
    lonlat: tuple = (0.0, 0.0)
    basin_area_km2: float = 0.0
    population_exposed: int = 0
    time_to_arrival_min: float = float("nan")   # flood only, from route()
    arrival_low_min: float = float("nan")       # the honest band, not a minute
    arrival_high_min: float = float("nan")


@dataclass
class Forecast:
    api_version: str = API_VERSION
    provenance: Provenance = field(default_factory=Provenance)
    grid: GridSpec = field(default_factory=GridSpec)
    lead_minutes: tuple = LEAD_MINUTES
    fields: list = field(default_factory=list)     # HazardField
    alerts: list = field(default_factory=list)     # Alert

    def to_dict(self) -> dict:
        return {
            "api_version": self.api_version,
            "provenance": asdict(self.provenance),
            "grid": asdict(self.grid),
            "lead_minutes": list(self.lead_minutes),
            "fields": [asdict(f) for f in self.fields],
            "alerts": [asdict(a) for a in self.alerts],
        }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
#
#   GET /api/v1/health          -> {"status", "model_version", "registration_gate"}
#   GET /api/v1/grid            -> GridSpec
#   GET /api/v1/forecast        -> Forecast   (all hazards, all leads)
#         ?hazard=thunderstorm  -> one hazard
#         ?lead=180             -> one lead time
#   GET /api/v1/alerts          -> {"alerts": [...]}   the feed
#         ?severity=warning
#   GET /api/v1/cell?lon=&lat=  -> the inspector: every hazard, every lead,
#                                  at one location, WITH the basin it sits in
#                                  and the terrain that decides its flood risk
#
# The scrubber is client-side over `lead_minutes`; the server returns all
# leads in one response so scrubbing costs no round trips.


def stub_predict(shape=(912, 864), n_stations=48, n_basins=64, seed=0):
    """Arrays of the right shape, so the dashboard can be built now.

    Deliberately labelled `model_version="stub"` and `calibrated=False` in the
    provenance, so a stub forecast can never be mistaken for a real one in a
    screenshot or a demo.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    h, w = shape
    small = (h // 16, w // 16)
    now = datetime.now(timezone.utc).replace(microsecond=0)

    fields = []
    for lead in LEAD_MINUTES:
        base = rng.random(small)
        for _ in range(4):
            base = (base + np.roll(base, 1, 0) + np.roll(base, 1, 1)) / 3.0
        grid = np.kron(base, np.ones((16, 16)))[:h, :w]
        grid = np.clip((grid - grid.min()) / (float(grid.max() - grid.min()) or 1), 0, 1) ** 2
        fields.append(encode_grid("thunderstorm", lead, grid))
        fields.append(HazardField(
            "cloudburst", "point", lead,
            values=(rng.random(n_stations) ** 4).round(4).tolist(),
            element_ids=[f"AWS{i:03d}" for i in range(n_stations)],
            element_lonlat=[[float(68 + 30 * rng.random()),
                             float(8 + 28 * rng.random())] for _ in range(n_stations)]))
        fields.append(HazardField(
            "flash_flood", "basin", lead,
            values=(rng.random(n_basins) ** 3).round(4).tolist(),
            element_ids=[f"B{i:04d}" for i in range(n_basins)],
            element_lonlat=[[float(70 + 26 * rng.random()),
                             float(10 + 24 * rng.random())] for _ in range(n_basins)]))

    prov = Provenance(model_version="stub", cache_fingerprint="86fae373df9155e7",
                      issued_at=now.isoformat(), satellite="3RIMG",
                      scan_time_utc=now.isoformat(),
                      registration_gate="NOT_RUN", calibrated=False,
                      notes="synthetic fields; not a forecast")
    return Forecast(provenance=prov, fields=fields,
                    alerts=build_alerts(fields, now))


def encode_grid(hazard: str, lead_minutes: int, grid, downsample: int = GRID_DOWNSAMPLE):
    """(H, W) floats in [0, 1] -> a HazardField carrying base64 uint8."""
    import base64

    import numpy as np

    g = np.asarray(grid, dtype=np.float64)
    d = max(1, int(downsample))
    h, w = g.shape
    hh, ww = h // d, w // d
    # Block MAXIMUM, not mean. Downsampling a hazard probability by averaging
    # dilutes exactly the isolated high-probability cell the product exists
    # to show -- a 4 km cloudburst signal averaged over 16 km reads as calm.
    small = g[:hh * d, :ww * d].reshape(hh, d, ww, d).max(axis=(1, 3))
    u8 = np.clip(small * 255.0, 0, 255).astype(np.uint8)
    return HazardField(hazard, "grid", lead_minutes,
                       grid_b64=base64.b64encode(u8.tobytes()).decode("ascii"),
                       grid_shape=(hh, ww), grid_downsample=d)


def build_alerts(fields, now=None, thresholds=None, far=None):
    """Fields -> the alert feed.

    Thresholds are arguments, not constants, and each alert carries the FAR
    of the operating point it came from. An alert that cannot say what it
    costs is a colour on a map.
    """
    now = now or datetime.now(timezone.utc).replace(microsecond=0)
    thresholds = thresholds or {"advisory": 0.30, "watch": 0.50, "warning": 0.70}
    far = far or {"advisory": 0.95, "watch": 0.85, "warning": 0.997}

    out = []
    for f in fields:
        if f.geometry == "grid":
            continue          # grid alerts are drawn from the map, not listed
        for i, v in enumerate(f.values):
            sev = None
            for s in ("warning", "watch", "advisory"):
                if v >= thresholds[s]:
                    sev = s
                    break
            if sev is None:
                continue
            eid = f.element_ids[i] if i < len(f.element_ids) else str(i)
            lonlat = tuple(f.element_lonlat[i]) if i < len(f.element_lonlat) else (0.0, 0.0)
            out.append(Alert(
                alert_id=f"{f.hazard}:{eid}:{f.lead_minutes}",
                hazard=f.hazard, severity=sev, probability=float(v),
                threshold=thresholds[sev], far_at_threshold=far[sev],
                valid_from=now.isoformat(),
                valid_to=(now + timedelta(minutes=f.lead_minutes)).isoformat(),
                lead_minutes=f.lead_minutes, element_id=eid, lonlat=lonlat))
    out.sort(key=lambda a: (-a.probability, a.lead_minutes))
    return out
