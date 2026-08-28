"""Scoring all three MTL heads together.

The problem statement's architecture is one shared backbone branching into
thunderstorm, cloudburst and flash-flood heads. That creates a failure mode
a single-head scorecard cannot see: **one head improves while another
quietly degrades**. The shared backbone reallocates capacity toward
whichever task dominates the gradient -- and with base rates differing by
two orders of magnitude, that will be the thunderstorm head. You can watch
a run's headline number improve for six epochs while the cloudburst head,
the one the project actually exists for, gets steadily worse.

This module scores every head through the same code path and makes that
divergence explicit.

A note on aggregation
---------------------
There is deliberately no "overall score" that averages the three heads.
Thunderstorms occur at ~1e-2 and cloudbursts at ~1e-4; a mean of their CSIs
is a number with no meaning, and it would hide exactly the failure this
module exists to catch. Where a single number is unavoidable (checkpoint
selection) `worst_head` is offered instead -- the conservative choice,
which cannot be gamed by one head carrying the others.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

from .config import EvalConfig
from .core import EvaluationResult, evaluate


@dataclass
class MultiHazardResult:
    hazards: dict[str, EvaluationResult] = field(default_factory=dict)

    def __getitem__(self, k: str) -> EvaluationResult:
        return self.hazards[k]

    @property
    def names(self) -> list[str]:
        return list(self.hazards)

    def to_dict(self) -> dict:
        return {k: v.to_dict() for k, v in self.hazards.items()}

    def to_json(self, path: str) -> None:
        import json
        with open(path, "w") as fh:
            json.dump(self.to_dict(), fh, indent=2, default=float)

    # --- single numbers, chosen carefully ---------------------------------
    def headline(self, metric: str = "csi") -> dict[str, float]:
        return {k: v.pooled["headline"].get(metric, np.nan)
                if metric in v.pooled["headline"]
                else v.pooled["probabilistic"].get(metric, np.nan)
                for k, v in self.hazards.items()}

    def worst_head(self, metric: str = "csi") -> tuple[str, float]:
        """The conservative checkpoint metric: the head doing worst.

        Optimising this means a run cannot win by sacrificing the rare
        hazards for the common one.
        """
        vals = {k: v for k, v in self.headline(metric).items() if np.isfinite(v)}
        if not vals:
            return ("", float("nan"))
        k = min(vals, key=vals.get)
        return (k, vals[k])

    # --- reporting ---------------------------------------------------------
    def summary_table(self) -> str:
        lines = ["=" * 78, "MULTI-HAZARD SCORECARD", "=" * 78, ""]
        hdr = (f"{'hazard':>14} {'base rate':>11} {'events':>9} "
               f"{'CSI':>7} {'POD':>7} {'FAR':>7} {'BSS':>8}")
        lines += [hdr, "-" * len(hdr)]
        for name, r in self.hazards.items():
            h = r.pooled["headline"]
            events = h["hits"] + h["misses"]
            lines.append(f"{name:>14} {r.pooled['base_rate']:>11.2e} {events:>9d} "
                         f"{h['csi']:>7.3f} {h['pod']:>7.3f} {h['far']:>7.3f} "
                         f"{r.pooled['probabilistic']['bss']:>8.3f}")
        w, v = self.worst_head()
        lines += ["", f"worst head: {w} (CSI {v:.3f}) <- checkpoint on this, "
                      f"not on the mean"]

        thin = [n for n, r in self.hazards.items()
                if (r.pooled["headline"]["hits"] + r.pooled["headline"]["misses"]) < 100]
        if thin:
            lines.append(f"!!  too few events to trust: {', '.join(thin)} "
                         f"-- attach bootstrap CIs before comparing models")
        lines.append("")
        for name, r in self.hazards.items():
            lines += [f"--- {name} " + "-" * (70 - len(name)), r.summary_table(), ""]
        return "\n".join(lines)

    def compare(self, previous: "MultiHazardResult", metric: str = "csi",
                tol: float = 0.005) -> str:
        """Did any head regress since the last epoch/run?

        Run this every epoch. It is the cheapest possible guard against the
        MTL failure mode described in the module docstring.
        """
        now, before = self.headline(metric), previous.headline(metric)
        lines = [f"head-by-head change in {metric.upper()}:"]
        regressed = []
        for k in now:
            if k not in before:
                continue
            d = now[k] - before[k]
            flag = ""
            if d < -tol:
                flag = "  <-- REGRESSED"
                regressed.append(k)
            elif d > tol:
                flag = "  improved"
            lines.append(f"    {k:>14}  {before[k]:.3f} -> {now[k]:.3f}  "
                         f"({d:+.3f}){flag}")
        if regressed:
            lines.append(f"!!  {len(regressed)} head(s) got worse: "
                         f"{', '.join(regressed)}. The shared backbone is "
                         f"reallocating capacity -- check task loss weights.")
        else:
            lines.append("    no head regressed")
        return "\n".join(lines)


def evaluate_multi(
    preds: Mapping[str, np.ndarray],
    obs: Mapping[str, np.ndarray],
    configs: Mapping[str, EvalConfig] | EvalConfig | None = None,
    masks: Mapping[str, np.ndarray] | None = None,
    lead_minutes: Sequence[float] | None = None,
    meta: dict | None = None,
) -> MultiHazardResult:
    """Score every head. Each hazard keeps its own config.

    Per-hazard configs matter: the alerting threshold and the label
    definition that make sense for a thunderstorm are not the ones that make
    sense for a cloudburst (IMD's ~100 mm/hr over 20-30 km^2). Passing a
    single shared EvalConfig is allowed but is usually the wrong call.
    """
    if set(preds) != set(obs):
        raise ValueError(f"hazard keys differ: preds {sorted(preds)} vs obs {sorted(obs)}")
    if not preds:
        raise ValueError("no hazards given")

    out = MultiHazardResult()
    for name in preds:
        if isinstance(configs, Mapping):
            cfg = configs.get(name, EvalConfig(name=name))
        else:
            cfg = configs or EvalConfig(name=name)
        out.hazards[name] = evaluate(
            preds[name], obs[name], cfg,
            mask=(masks or {}).get(name),
            lead_minutes=lead_minutes,
            meta=(meta or {}) | {"hazard": name},
        )
    return out
