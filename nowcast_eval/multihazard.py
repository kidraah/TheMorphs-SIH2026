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

Which metric to take the worst OF
---------------------------------
Not CSI. CSI is not comparable across heads: it falls with the base rate
whatever the forecast quality (a fixed-quality forecast scores 0.55 at a
1e-1 base rate and 0.0016 at 1e-4), so `min` over CSI would pick the
rarest hazard every single time regardless of how well any head performs.
That is a constant, not a signal.

Use SEDI. It is base-rate independent, which is exactly the property that
makes a cross-head minimum mean something. CSI stays worth reporting for
the thunderstorm head, where the base rate supports it -- but it is a
reporting metric there, not a selection metric.

And select on the CI LOWER BOUND, not the point estimate. At a 2e-4 base
rate the point estimate is noise-dominated, so choosing epochs on it is
choosing on luck: the checkpoint that wins is the one whose validation set
happened to break its way. The lower bound asks the stricter question --
what can we actually defend? -- and cannot be won by a lucky draw.
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
        out = {}
        for k, v in self.hazards.items():
            if metric in v.pooled["headline"]:
                out[k] = v.pooled["headline"][metric]
            else:
                out[k] = v.pooled["probabilistic"].get(metric, np.nan)
        return out

    def worst_head(self, metric: str = "sedi") -> tuple[str, float]:
        """The head doing worst, on a base-rate-independent metric.

        Defaults to SEDI. Passing metric="csi" here is almost always a
        mistake -- see the module docstring: CSI's base-rate dependence
        makes the minimum select the rarest hazard by construction.
        """
        vals = {k: v for k, v in self.headline(metric).items() if np.isfinite(v)}
        if not vals:
            return ("", float("nan"))
        k = min(vals, key=vals.get)
        return (k, vals[k])

    def worst_head_lower_bound(self, metric: str = "sedi") -> tuple[str, float]:
        """The checkpoint metric: worst head's LOWER confidence bound.

        Requires `attach_confidence_intervals` to have been run on each
        head first. Selecting on the point estimate of a rare-event score
        is selecting on sampling noise -- the winning epoch is the one
        whose validation draw was luckiest, which does not generalise.

        Raises rather than silently falling back to the point estimate: a
        checkpoint metric that quietly changes meaning is worse than one
        that fails loudly.
        """
        missing = [k for k, v in self.hazards.items() if "ci" not in v.pooled]
        if missing:
            raise ValueError(
                f"no confidence intervals on: {', '.join(missing)}. Run "
                f"attach_confidence_intervals(result[hazard], pred, obs, cfg) "
                f"for each head before checkpointing on the lower bound."
            )
        vals, undefined = {}, []
        for k, v in self.hazards.items():
            lo = v.pooled["ci"]["lo"].get(metric, np.nan)
            if lo is None or not np.isfinite(lo):
                undefined.append(k)
            else:
                vals[k] = float(lo)

        # A head whose score is undefined must rank WORST, never be skipped.
        # Dropping it silently is the exact failure this module exists to
        # prevent: selection would report a healthy number while blind to a
        # head -- and the blind one is always the rarest, most important
        # hazard, because that is where the evidence runs out first.
        if undefined:
            return (undefined[0], float("-inf"))
        if not vals:
            return ("", float("nan"))
        k = min(vals, key=vals.get)
        return (k, vals[k])

    # --- reporting ---------------------------------------------------------
    def summary_table(self) -> str:
        lines = ["=" * 78, "MULTI-HAZARD SCORECARD", "=" * 78, ""]
        hdr = (f"{'hazard':>14} {'base rate':>11} {'events':>9} "
               f"{'SEDI [95% CI]':>24} {'CSI':>7} {'POD':>7} {'FAR':>7} {'BSS':>8}")
        lines += [hdr, "-" * len(hdr)]
        for name, r in self.hazards.items():
            h = r.pooled["headline"]
            events = h["hits"] + h["misses"]
            ci = r.pooled.get("ci")
            if ci and np.isfinite(ci["lo"].get("sedi", np.nan)):
                sedi = (f"{h['sedi']:.3f} "
                        f"[{ci['lo']['sedi']:+.3f},{ci['hi']['sedi']:+.3f}]")
            else:
                sedi = f"{h['sedi']:.3f}"
            lines.append(f"{name:>14} {r.pooled['base_rate']:>11.2e} {events:>9d} "
                         f"{sedi:>24} "
                         f"{h['csi']:>7.3f} {h['pod']:>7.3f} {h['far']:>7.3f} "
                         f"{r.pooled['probabilistic']['bss']:>8.3f}")

        lines.append("")
        try:
            w, v = self.worst_head_lower_bound("sedi")
            if v == float("-inf"):
                lines.append(f"CHECKPOINT BLOCKED: {w} SEDI is undefined -- too few "
                             f"events to measure it. Do not checkpoint while a head "
                             f"is unmeasurable; widen the test set or lower the "
                             f"threshold for that head.")
            else:
                lines.append(f"CHECKPOINT ON: {w} SEDI lower bound = {v:.3f}")
            wp, vp = self.worst_head("sedi")
            if np.isfinite(vp):
                lines.append(f"    (point estimate would pick {wp} at {vp:.3f} -- "
                             f"noise-dominated at these base rates)")
        except ValueError:
            w, v = self.worst_head("sedi")
            lines.append(f"worst head: {w} (SEDI {v:.3f}) -- point estimate only; "
                         f"attach CIs and checkpoint on the lower bound instead")
        lines.append("    CSI shown for reporting; it is not comparable across "
                     "heads (base-rate dependent)")

        thin = [n for n, r in self.hazards.items()
                if (r.pooled["headline"]["hits"] + r.pooled["headline"]["misses"]) < 100]
        if thin:
            lines.append(f"!!  too few events to trust: {', '.join(thin)} "
                         f"-- attach bootstrap CIs before comparing models")
        lines.append("")
        for name, r in self.hazards.items():
            lines += [f"--- {name} " + "-" * (70 - len(name)), r.summary_table(), ""]
        return "\n".join(lines)

    def compare(self, previous: "MultiHazardResult", metric: str = "sedi",
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
