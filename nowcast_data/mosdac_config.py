"""Estimate what a MOSDAC config will actually match, and refuse the bad ones.

Instance 11, at a new layer. A config that returns 179,134 files and 69.70 TB
is syntactically valid JSON, semantically catastrophic, and passed every
check we had -- because we had none. The only thing between it and a 70 TB
pull was the download client happening to ask for confirmation.

    "startTime": "", "endTime": ""

Empty is not "unset, please ignore". To the search API it is an unbounded
range, so it matched the entire 3RIMG archive from 2016-10-11 to now. The
dates were written into a `_one_job_per_date` key that the client never
reads, on the assumption it would iterate -- it takes ONE start/end per
file. The generator produced something that looked like six jobs and was one
job over nine years.

The shape is familiar: a value that decodes to something plausible and
catastrophically wrong, with no symptom until it is too late. It differs
from the data instances only in that the artefact was a config, not a
raster, and nothing in the tooling looked at it.

So: estimate before writing, and REFUSE above a threshold rather than
discovering the size at the download prompt.

VALIDATION. The estimator is checked against the one real observation we
have: the config that returned 179,134 granules. See
tests/test_mosdac_config.py -- if the model drifts, that test fails.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

# Scans per day per dataset. 3RIMG/3DIMG are 30-minute full-disk imagers.
SCANS_PER_DAY = {
    "3RIMG_L1B_STD": 48, "3DIMG_L1B_STD": 48, "3SIMG_L1B_STD": 48,
    "3RIMG_L1C_ASIA_MER": 48, "3DIMG_L1C_ASIA_MER": 48,
}
DEFAULT_SCANS_PER_DAY = 48

# Measured on 702 delivered files: 3RIMG mean 448.0 MB, 3DIMG 434.4 MB.
MEAN_GRANULE_MB = {"3RIMG_L1B_STD": 448.0, "3DIMG_L1B_STD": 434.4}
DEFAULT_GRANULE_MB = 440.0

# First granule of each archive; the end is "today" unless stated.
DATASET_START = {
    "3RIMG_L1B_STD": date(2016, 10, 11),
    "3DIMG_L1B_STD": date(2013, 9, 1),
    "3SIMG_L1B_STD": date(2024, 6, 18),
}
DEFAULT_DATASET_START = date(2013, 9, 1)

# A gate or format-check config wants a handful. An archive slice wants a
# fortnight. Anything above this is a mistake until proven otherwise.
MAX_GRANULES_DEFAULT = 200


class ConfigTooBroad(ValueError):
    """The config would match more granules than allowed."""


@dataclass
class Estimate:
    granules: int
    days: int
    gb: float
    dataset: str
    start: str
    end: str
    unbounded: bool
    reasoning: str

    def report(self) -> str:
        return (f"{self.dataset} {self.start}..{self.end} "
                f"({self.days:,} days) -> ~{self.granules:,} granules, "
                f"~{self.gb:,.0f} GB ({self.gb/1024:.2f} TB)"
                + ("  [UNBOUNDED RANGE]" if self.unbounded else ""))


def _parse(s, default):
    if not s or not str(s).strip():
        return default, True
    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%d%b%Y"):
        try:
            return datetime.strptime(str(s).strip(), fmt).date(), False
        except ValueError:
            continue
    raise ValueError(f"unparseable date {s!r}")


def estimate(cfg: dict, today: date | None = None) -> Estimate:
    """How many granules would this config's search match?"""
    sp = cfg.get("search_parameters", {}) or {}
    ds = sp.get("datasetId", "") or "unknown"
    today = today or date.today()

    start, s_unbounded = _parse(sp.get("startTime"),
                                DATASET_START.get(ds, DEFAULT_DATASET_START))
    end, e_unbounded = _parse(sp.get("endTime"), today)
    unbounded = s_unbounded or e_unbounded

    days = max((end - start).days + 1, 0)
    per_day = SCANS_PER_DAY.get(ds, DEFAULT_SCANS_PER_DAY)
    n = days * per_day

    # An explicit `count` caps the search.
    count = sp.get("count")
    if count not in (None, "", 0):
        try:
            n = min(n, int(count))
        except (TypeError, ValueError):
            pass

    mb = MEAN_GRANULE_MB.get(ds, DEFAULT_GRANULE_MB)
    why = f"{days:,} days x {per_day} scans/day"
    if unbounded:
        why += (" -- start and/or end were EMPTY, which the search API reads "
                "as unbounded, not as 'ignore this field'")
    return Estimate(n, days, n * mb / 1024, ds, str(start), str(end),
                    unbounded, why)


def validate(cfg: dict, max_granules: int = MAX_GRANULES_DEFAULT,
             today: date | None = None) -> Estimate:
    """Raise ConfigTooBroad if this config would match too much."""
    est = estimate(cfg, today)
    sp = cfg.get("search_parameters", {}) or {}

    ignored = [k for k in cfg
               if k.startswith("_") and ("job" in k.lower() or "date" in k.lower()
                                         or "scene" in k.lower())]
    if est.unbounded:
        raise ConfigTooBroad(
            f"startTime/endTime are empty, so this matches the WHOLE archive: "
            f"{est.report()}.\n"
            f"Empty is an unbounded range, not an unset field. "
            + (f"Dates in {ignored} are comments -- the client reads only "
               f"search_parameters. One date range per file; write one file "
               f"per date." if ignored else
               "Set both startTime and endTime."))
    if est.granules > max_granules:
        raise ConfigTooBroad(
            f"this config would match ~{est.granules:,} granules "
            f"(~{est.gb:,.0f} GB), above the {max_granules:,} limit.\n"
            f"  {est.reasoning}\n"
            f"  Split it into smaller date ranges, or raise max_granules "
            f"deliberately if a bulk slice is really intended.")
    return est


def write(path, cfg: dict, max_granules: int = MAX_GRANULES_DEFAULT,
          today: date | None = None) -> Estimate:
    """Validate, THEN write. A too-broad config never reaches disk."""
    est = validate(cfg, max_granules, today)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(cfg, indent=2))
    return est


def audit(directory, max_granules: int = MAX_GRANULES_DEFAULT,
          today: date | None = None) -> list[dict]:
    """Estimate every config in a directory. For checking what already exists."""
    out = []
    for p in sorted(Path(directory).glob("*.json")):
        try:
            cfg = json.loads(p.read_text())
        except Exception as e:
            out.append({"file": p.name, "error": f"unparseable: {e}"})
            continue
        if "search_parameters" not in cfg:
            continue
        est = estimate(cfg, today)
        out.append({"file": p.name, "granules": est.granules, "gb": est.gb,
                    "unbounded": est.unbounded,
                    "over_limit": est.granules > max_granules,
                    "report": est.report()})
    return out
