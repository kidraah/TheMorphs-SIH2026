"""The entry point: hand it predictions and truth, get back a scorecard.

The harness has no skill of its own. It is a ruler. Everything here is
deterministic given (pred, obs, config) -- there is no fitting, no
randomness, no state. That is what makes the tests in tests/ able to
prove it correct.

Data contract
-------------
    pred : (N, L, H, W) float, probabilities in [0, 1]
    obs  : (N, L, H, W) truth -- boolean, or continuous with
           `config.obs_threshold` set to binarise it
    mask : (N, L, H, W) or broadcastable bool, True = valid

    N = samples (event windows / forecast issue times)
    L = lead times (e.g. 12 steps of 30 min out to 6 h)
    H, W = grid

Scores are broken out per lead time because a single pooled number hides
the only thing that matters: skill decays with lead time, and the whole
claim of the project is about where it is still useful at 2-6 hours.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from .config import EvalConfig
from .contingency import contingency
from .fss import fss, useful_scale_threshold
from .probabilistic import probabilistic_scores


@dataclass
class LeadTimeScores:
    lead_index: int
    lead_minutes: float | None
    base_rate: float
    contingency: list[dict]          # one row per swept threshold
    headline: dict                   # the row at config.headline_threshold
    fss: list[dict]                  # one row per (threshold, neighbourhood)
    probabilistic: dict
    ci: dict | None = None           # filled by bootstrap.attach_confidence_intervals

    def to_dict(self) -> dict:
        return {
            "lead_index": self.lead_index,
            "lead_minutes": self.lead_minutes,
            "base_rate": self.base_rate,
            "contingency": self.contingency,
            "headline": self.headline,
            "fss": self.fss,
            "probabilistic": self.probabilistic,
            "ci": self.ci,
        }


@dataclass
class EvaluationResult:
    config: dict
    per_lead: list[LeadTimeScores]
    pooled: dict
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "config": self.config,
            "meta": self.meta,
            "pooled": self.pooled,
            "per_lead": [l.to_dict() for l in self.per_lead],
        }

    def to_json(self, path: str) -> None:
        with open(path, "w") as fh:
            json.dump(self.to_dict(), fh, indent=2, default=float)

    def summary_table(self) -> str:
        """The thing you actually read after a training run."""
        cfg = self.config
        lines = []
        lines.append(f"eval: {cfg['name']}   headline threshold p>={cfg['headline_threshold']}")
        if cfg.get("geometry", "grid") == "point":
            lines.append("geometry: point (station locations)")
        elif cfg.get("geometry", "grid") == "basin":
            lines.append(f"geometry: basin (sub-basin polygons, "
                         f"{cfg.get('basin_weighting', 'area')}-weighted)")
        else:
            lines.append(f"grid {cfg['grid_km']} km   "
                         f"neighbourhoods {list(cfg['neighborhood_km'])} km")
        lines.append("")
        head = f"{'lead':>8} {'base rate':>11} {'POD':>7} {'FAR':>7} {'CSI':>7} {'bias':>7} {'BSS':>8}"
        point = cfg.get("geometry", "grid") in ("point", "basin")
        fss_cols = [] if point else [f"FSS@{int(k)}km" for k in cfg["neighborhood_km"]]
        head += "".join(f"{c:>10}" for c in fss_cols)
        lines.append(head)
        lines.append("-" * len(head))

        for l in self.per_lead:
            h = l.headline
            lead = f"{l.lead_minutes:g}m" if l.lead_minutes is not None else f"#{l.lead_index}"
            row = (f"{lead:>8} {l.base_rate:>11.2e} {h['pod']:>7.3f} {h['far']:>7.3f} "
                   f"{h['csi']:>7.3f} {h['frequency_bias']:>7.3f} "
                   f"{l.probabilistic['bss']:>8.3f}")
            for km in ([] if point else cfg["neighborhood_km"]):
                m = next((f for f in l.fss
                          if f["neighborhood_km"] == km
                          and f["threshold"] == cfg["headline_threshold"]), None)
                row += f"{(m['fss'] if m else float('nan')):>10.3f}"
            lines.append(row)

        lines.append("")
        lines.append(f"pooled base rate {self.pooled['base_rate']:.3e}  "
                     f"CSI {self.pooled['headline']['csi']:.3f}  "
                     f"BSS {self.pooled['probabilistic']['bss']:.3f}")
        if point:
            lines.append("geometry is not a grid: FSS omitted -- element order is not "
                         "a spatial neighbourhood")
        else:
            lines.append(f"'useful' FSS line for this base rate: "
                         f"{useful_scale_threshold(self.pooled['base_rate']):.3f}")

        # Confidence intervals, if attached. Printed as a separate block so the
        # point estimates never appear without their uncertainty beside them.
        ci = self.pooled.get("ci")
        if ci:
            lines.append("")
            lines.append(f"pooled 95% CI over {ci['n_samples']} forecast cases, "
                         f"{ci['n_events']} observed events "
                         f"({ci['n_boot']} bootstrap replicates)")
            for m in ("csi", "pod", "far", "bss"):
                pt, lo, hi = ci["point"].get(m), ci["lo"].get(m), ci["hi"].get(m)
                if pt is not None and np.isfinite(pt):
                    lines.append(f"    {m.upper():>4}  {pt:.3f}  [{lo:.3f}, {hi:.3f}]")
            for w in ci["warnings"]:
                lines.append(f"    !!  {w}")
        return "\n".join(lines)


def _binarise_obs(obs: np.ndarray, cfg: EvalConfig) -> np.ndarray:
    if cfg.obs_threshold is None:
        return obs
    return (np.asarray(obs, dtype=np.float64) >= cfg.obs_threshold).astype(np.float64)


def _score_block(pred, obs, mask, cfg: EvalConfig, weights=None) -> dict:
    """All scores for one slab of (pred, obs). Used per-lead and pooled."""
    tables = [contingency(pred, obs, t, mask, weights).to_dict()
              for t in cfg.thresholds]
    headline = contingency(pred, obs, cfg.headline_threshold, mask,
                           weights).to_dict()

    # Point geometry has no spatial neighbourhood: stations are irregularly
    # spaced and their array order carries no distance information, so a
    # neighbourhood filter over it would be meaningless. FSS is omitted
    # rather than computed on nonsense.
    fss_rows = []
    if cfg.has_neighborhood:
        for size, km in zip(cfg.neighborhood_pixels(), cfg.neighborhood_km):
            for t in cfg.thresholds:
                fss_rows.append(fss(pred, obs, t, size, mask, grid_km=km).to_dict()
                                | {"neighborhood_km": km})

    prob = probabilistic_scores(pred, obs, cfg.n_reliability_bins, mask).to_dict()
    return {
        "contingency": tables,
        "headline": headline,
        "fss": fss_rows,
        "probabilistic": prob,
        "base_rate": prob["base_rate"],
    }


def evaluate(
    pred: np.ndarray,
    obs: np.ndarray,
    config: EvalConfig | None = None,
    mask: np.ndarray | None = None,
    lead_minutes: Sequence[float] | None = None,
    meta: dict[str, Any] | None = None,
    weights: np.ndarray | None = None,
) -> EvaluationResult:
    """Score a set of probabilistic nowcasts. See module docstring for shapes."""
    cfg = config or EvalConfig()
    pred = np.asarray(pred, dtype=np.float64)
    obs = _binarise_obs(np.asarray(obs), cfg)

    if pred.shape != obs.shape:
        raise ValueError(f"shape mismatch: pred {pred.shape} vs obs {obs.shape}")
    if cfg.is_point or cfg.is_basin:
        if pred.ndim != 3:
            shape, unit = (("(N, L, S)", "S station locations") if cfg.is_point
                           else ("(N, L, B)", "B sub-basin polygons"))
            raise ValueError(
                f"geometry={cfg.geometry!r} expects {shape} with {unit}, "
                f"got {pred.shape}")
    elif pred.ndim != 4:
        raise ValueError(
            f"expected (N, L, H, W), got {pred.shape}. For station-point truth "
            f"(e.g. IMD AWS/ARG gauges) set EvalConfig(geometry='point') and "
            f"pass (N, L, S) -- do NOT reshape stations into a 1 x S grid, "
            f"which would make FSS average over station index.")
    if np.isfinite(pred).any() and (np.nanmin(pred) < 0 or np.nanmax(pred) > 1):
        raise ValueError("pred must be probabilities in [0, 1]")

    # Basin geometry REQUIRES an explicit weighting decision. Defaulting to
    # equal weights here would reproduce, silently, the exact error the
    # geometry field exists to prevent: basins are not interchangeable units.
    if cfg.is_basin:
        if cfg.basin_weighting == "equal":
            weights = None
        elif weights is None:
            raise ValueError(
                f"geometry='basin' with basin_weighting="
                f"{cfg.basin_weighting!r} needs `weights` -- one non-negative "
                f"value per basin (area in km^2, or population), broadcastable "
                f"to {pred.shape}. Sub-basin areas differ by orders of "
                f"magnitude, so counting each basin once measures skill per "
                f"basin, not per unit of land. Pass "
                f"EvalConfig(basin_weighting='equal') if that is genuinely "
                f"what you want.")
    elif weights is not None:
        raise ValueError(f"weights are only meaningful for geometry='basin'; "
                         f"got geometry={cfg.geometry!r}")
    if weights is not None:
        weights = np.broadcast_to(np.asarray(weights, dtype=np.float64),
                                  pred.shape)

    if mask is not None:
        mask = np.broadcast_to(np.asarray(mask, dtype=bool), pred.shape)
    if not cfg.drop_masked and mask is not None:
        # Explicitly requested: treat masked cells as observed no-event.
        obs = np.where(mask, obs, 0.0)
        mask = None

    n_lead = pred.shape[1]
    if lead_minutes is not None and len(lead_minutes) != n_lead:
        raise ValueError(f"lead_minutes has {len(lead_minutes)} entries, pred has {n_lead}")

    per_lead = []
    for li in range(n_lead):
        m = mask[:, li] if mask is not None else None
        blk = _score_block(pred[:, li], obs[:, li], m, cfg,
                           weights[:, li] if weights is not None else None)
        per_lead.append(LeadTimeScores(
            lead_index=li,
            lead_minutes=float(lead_minutes[li]) if lead_minutes is not None else None,
            base_rate=blk["base_rate"],
            contingency=blk["contingency"],
            headline=blk["headline"],
            fss=blk["fss"],
            probabilistic=blk["probabilistic"],
        ))

    pooled = _score_block(pred.reshape(-1, *pred.shape[2:]),
                          obs.reshape(-1, *obs.shape[2:]),
                          mask.reshape(-1, *mask.shape[2:]) if mask is not None else None,
                          cfg,
                          weights.reshape(-1, *weights.shape[2:])
                          if weights is not None else None)
    pooled["geometry"] = cfg.geometry
    if cfg.is_basin:
        pooled["basin_weighting"] = cfg.basin_weighting

    return EvaluationResult(
        config=cfg.to_dict(),
        per_lead=per_lead,
        pooled=pooled,
        meta=meta or {},
    )


def evaluate_predict_fn(
    predict_fn: Callable[[np.ndarray], np.ndarray],
    dataset: Iterable[tuple[np.ndarray, np.ndarray]],
    config: EvalConfig | None = None,
    lead_minutes: Sequence[float] | None = None,
    meta: dict[str, Any] | None = None,
    weights: np.ndarray | None = None,
) -> EvaluationResult:
    """Convenience wrapper: run a model over a dataset, then score it.

    `predict_fn` takes an input tensor and returns (L, H, W) probabilities.
    `dataset` yields (inputs, truth) pairs. This is the signature the
    training loop's validation hook uses, so the model is scored by exactly
    the same code path as the final report -- no second implementation to
    drift out of sync.
    """
    preds, obss = [], []
    for x, y in dataset:
        p = np.asarray(predict_fn(x), dtype=np.float64)
        preds.append(p)
        obss.append(np.asarray(y))
    return evaluate(np.stack(preds), np.stack(obss), config,
                    lead_minutes=lead_minutes, meta=meta)
