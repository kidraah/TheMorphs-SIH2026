"""Known gaps: absent data whose absence is explained.

A known gap and missing data are not the same thing, and the harness has to
tell them apart in both directions:

    MISSING     we expected a granule and do not have it. That is a defect in
                the pull, and an audit should keep reporting it until it is
                fixed or explained.
    KNOWN GAP   we expected it, cannot have it, and know why. Scoring it as
                a miss penalises the model for a MOSDAC 500, and silently
                dropping it hides that the hindcast has a hole.

The failure this prevents is specific. A hindcast scored across an absent
slot has two wrong options and no right one unless the gap is declared:

  * treat the absent slot as "no event" -> the model is charged with a false
    alarm wherever it correctly forecast rain into the gap, and the base rate
    is diluted. Both flatter and penalise, in different metrics.
  * drop the case entirely -> the event looks shorter than it was, and a
    six-hour lead time scored over a five-hour event is not the claim being
    made.

Declaring it instead gives a third: MASK the slot, count it nowhere, and
report how much of the evaluation was masked so nobody reads a score without
knowing what it was computed over.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

DEFAULT_REGISTRY = Path(__file__).resolve().parents[1] / "configs" / "known_gaps.json"


@dataclass
class Gap:
    identifier: str
    scan_time_utc: str = ""
    reason: str = ""
    evidence: str = ""
    recoverable: bool = False
    event: str = ""
    impact: str = ""

    def __str__(self) -> str:
        return f"{self.identifier} ({self.scan_time_utc}): {self.reason}"


@dataclass
class GapRegistry:
    gaps: list = field(default_factory=list)

    @classmethod
    def load(cls, path=None) -> "GapRegistry":
        p = Path(path or DEFAULT_REGISTRY)
        if not p.exists():
            return cls()
        raw = json.loads(p.read_text())
        return cls([Gap(**{k: v for k, v in g.items() if k in Gap.__annotations__})
                    for g in raw.get("gaps", [])])

    def by_time(self) -> dict:
        return {g.scan_time_utc: g for g in self.gaps if g.scan_time_utc}

    def is_known(self, identifier=None, scan_time_utc=None) -> bool:
        for g in self.gaps:
            if identifier and g.identifier == identifier:
                return True
            if scan_time_utc and g.scan_time_utc == scan_time_utc:
                return True
        return False

    def unexplained(self, expected, present) -> list:
        """Granules absent WITHOUT an entry here. These are real shortfalls.

        The point of the registry is to make this list shrink to the genuinely
        broken, so that a non-empty result means something.
        """
        missing = set(expected) - set(present)
        return sorted(m for m in missing if not self.is_known(identifier=m))

    def report(self) -> str:
        if not self.gaps:
            return "no known gaps declared"
        lines = [f"{len(self.gaps)} known gap(s), each excluded from scoring:"]
        for g in self.gaps:
            lines.append(f"  {g}")
            if g.evidence:
                lines.append(f"      evidence: {g.evidence[:110]}")
            if g.recoverable:
                lines.append("      RECOVERABLE -- this should be re-pulled, "
                             "not carried as a gap")
        return "\n".join(lines)


def mask_known_gaps(mask, times, registry: GapRegistry | None = None):
    """Turn a validity mask False wherever a declared gap falls.

    `mask` is (N, L, ...) as `evaluate` takes it; `times` is one ISO
    timestamp per (case, lead) slot, shaped (N, L). Returns the new mask and
    how much was masked, because a score reported without the masked
    fraction is a score over an unstated support.
    """
    reg = registry or GapRegistry.load()
    m = np.array(mask, dtype=bool, copy=True)
    t = np.asarray(times, dtype=object)
    if t.shape != m.shape[:2]:
        raise ValueError(f"times {t.shape} must match the case/lead axes "
                         f"of mask {m.shape[:2]}")
    known = set(reg.by_time())
    hit = np.zeros(t.shape, dtype=bool)
    for i in range(t.shape[0]):
        for j in range(t.shape[1]):
            if str(t[i, j]) in known:
                hit[i, j] = True
                m[i, j] = False
    return m, {
        "slots_masked": int(hit.sum()),
        "slots_total": int(hit.size),
        "fraction_masked": float(hit.mean()),
        "gaps_hit": sorted({str(t[i, j]) for i, j in zip(*np.where(hit))}),
    }
