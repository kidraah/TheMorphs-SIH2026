"""Confidence intervals for rare-event scores.

Why this module exists
----------------------
A cloudburst test set may contain a few hundred event pixels in total.
CSI computed on 152 events and CSI computed on 150,000 events are printed
with identical authority -- three decimal places, no hint that one of them
is mostly noise. That is how a coincidence gets onto a slide.

These intervals make the difference visible. If your model scores
CSI 0.41 [0.22, 0.58] and the baseline scores 0.38 [0.20, 0.55], you have
not beaten the baseline, whatever the point estimates say.

Resampling unit: the SAMPLE (a forecast case), not the pixel
------------------------------------------------------------
Pixels inside one storm are massively spatially correlated -- neighbouring
cells are not independent draws. Resampling pixels would treat a single
storm as thousands of independent observations and produce intervals maybe
an order of magnitude too tight, which is worse than no interval at all
because it looks rigorous. Resampling whole forecast cases respects the
correlation structure, so this is a block bootstrap with the case as block.

How it stays fast
-----------------
Every metric here is a ratio of sums over cells. So we accumulate
per-sample sufficient statistics ONCE, then each bootstrap replicate is
just a re-sum of N numbers -- not a recomputation over the grid. The
replicates are therefore exact, not approximations, and 1000 of them cost
about as much as one extra evaluation.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import EvalConfig
from .fss import neighborhood_fractions


@dataclass
class PerSampleStats:
    """Sufficient statistics, one row per forecast case."""
    n: np.ndarray            # valid cells
    events: np.ndarray       # observed events
    sq_err: np.ndarray       # sum (p - o)^2   -> Brier
    hits: np.ndarray
    false_alarms: np.ndarray
    misses: np.ndarray
    fss_diff: dict = field(default_factory=dict)   # km -> sum (pf - po)^2
    fss_pf2: dict = field(default_factory=dict)    # km -> sum pf^2
    fss_po2: dict = field(default_factory=dict)    # km -> sum po^2

    @property
    def n_samples(self) -> int:
        return int(self.n.size)


def collect_stats(pred, obs, threshold, cfg: EvalConfig, mask=None) -> PerSampleStats:
    """One pass over the data.

    pred/obs are (N, ..., H, W). Axis 0 is the resampling unit -- the
    forecast case. EVERY trailing axis is summed into that case's row,
    including a lead-time axis if present: the 30-min and the 6-h frame of
    one storm are not independent observations, so they must be resampled
    together or the interval comes out too narrow.
    """
    pred = np.asarray(pred, dtype=np.float64)
    obs = np.asarray(obs, dtype=np.float64)
    valid = np.isfinite(pred) & np.isfinite(obs)
    if mask is not None:
        valid &= np.asarray(mask, dtype=bool)

    f = (pred >= threshold) & valid
    o = (obs > 0) & valid
    ax = tuple(range(1, pred.ndim))   # collapse everything below the case axis

    se = np.where(valid, (pred - (obs > 0)) ** 2, 0.0).sum(axis=ax)

    st = PerSampleStats(
        n=valid.sum(axis=ax).astype(np.float64),
        events=o.sum(axis=ax).astype(np.float64),
        sq_err=se,
        hits=(f & o).sum(axis=ax).astype(np.float64),
        false_alarms=(f & ~o & valid).sum(axis=ax).astype(np.float64),
        misses=(~f & o & valid).sum(axis=ax).astype(np.float64),
    )

    for size, km in zip(cfg.neighborhood_pixels(), cfg.neighborhood_km):
        pf = neighborhood_fractions(f, valid, size)
        po = neighborhood_fractions(o, valid, size)
        good = np.isfinite(pf) & np.isfinite(po) & valid
        a = np.where(good, pf, 0.0)
        b = np.where(good, po, 0.0)
        st.fss_diff[km] = ((a - b) ** 2).sum(axis=ax)
        st.fss_pf2[km] = (a ** 2).sum(axis=ax)
        st.fss_po2[km] = (b ** 2).sum(axis=ax)
    return st


def _metrics_from_sums(st: PerSampleStats, idx: np.ndarray, kms) -> dict:
    """Pool the resampled cases and form every ratio. All exact."""
    a = st.hits[idx].sum()
    b = st.false_alarms[idx].sum()
    c = st.misses[idx].sum()
    n = st.n[idx].sum()
    ev = st.events[idx].sum()
    se = st.sq_err[idx].sum()

    def div(x, y):
        return float(x) / float(y) if y > 0 else np.nan

    base = div(ev, n)
    bs = div(se, n)
    bs_clim = base * (1.0 - base) if np.isfinite(base) else np.nan
    out = {
        "csi": div(a, a + b + c),
        "pod": div(a, a + c),
        "far": div(b, a + b),
        "brier": bs,
        "bss": (np.nan if not np.isfinite(bs_clim) or bs_clim == 0
                else 1.0 - bs / bs_clim),
        "base_rate": base,
    }
    for km in kms:
        num = st.fss_diff[km][idx].sum()
        den = st.fss_pf2[km][idx].sum() + st.fss_po2[km][idx].sum()
        out[f"fss@{km:g}km"] = np.nan if den == 0 else 1.0 - float(num) / float(den)
    return out


@dataclass
class CIResult:
    point: dict          # metric -> point estimate on the real data
    lo: dict             # metric -> lower percentile
    hi: dict             # metric -> upper percentile
    n_events: int
    n_samples: int
    n_boot: int
    alpha: float
    warnings: list

    def to_dict(self) -> dict:
        return {
            "point": self.point, "lo": self.lo, "hi": self.hi,
            "n_events": self.n_events, "n_samples": self.n_samples,
            "n_boot": self.n_boot, "alpha": self.alpha,
            "warnings": self.warnings,
        }

    def fmt(self, metric: str, places: int = 3) -> str:
        p, lo, hi = self.point.get(metric), self.lo.get(metric), self.hi.get(metric)
        if p is None or not np.isfinite(p):
            return "nan"
        return f"{p:.{places}f} [{lo:.{places}f}, {hi:.{places}f}]"


def bootstrap_ci(
    pred: np.ndarray,
    obs: np.ndarray,
    config: EvalConfig | None = None,
    mask: np.ndarray | None = None,
    threshold: float | None = None,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> CIResult:
    """Percentile bootstrap CIs over forecast cases.

    pred/obs are (N, ..., H, W); axis 0 is the case and is the only axis
    resampled. Pass (N, L, H, W) to pool across lead times correctly.

    Returns the point estimate alongside the interval, plus explicit
    warnings when the sample is too thin for the interval itself to be
    trusted -- a CI computed from 8 cases is not a safeguard, it is a
    second thing that can mislead you.
    """
    cfg = config or EvalConfig()
    t = cfg.headline_threshold if threshold is None else threshold
    pred = np.asarray(pred, dtype=np.float64)
    if pred.ndim < 3:
        raise ValueError(f"expected (N, ..., H, W) with N the case axis, "
                         f"got {pred.shape}")

    st = collect_stats(pred, obs, t, cfg, mask)
    kms = list(cfg.neighborhood_km)
    N = st.n_samples
    all_idx = np.arange(N)

    point = _metrics_from_sums(st, all_idx, kms)
    n_events = int(st.events.sum())

    rng = np.random.default_rng(seed)
    reps = {k: np.empty(n_boot) for k in point}
    for i in range(n_boot):
        idx = rng.integers(0, N, size=N)
        m = _metrics_from_sums(st, idx, kms)
        for k, v in m.items():
            reps[k][i] = v

    lo, hi = {}, {}
    for k, v in reps.items():
        good = v[np.isfinite(v)]
        if good.size < n_boot * 0.5:
            lo[k], hi[k] = np.nan, np.nan
        else:
            lo[k] = float(np.percentile(good, 100 * alpha / 2))
            hi[k] = float(np.percentile(good, 100 * (1 - alpha / 2)))

    warns = []
    if N < 20:
        warns.append(f"only {N} forecast cases -- the bootstrap itself is unreliable "
                     f"below ~20 cases; widen the test set before trusting these bounds")
    if n_events < 100:
        warns.append(f"only {n_events} observed events -- point estimates are "
                     f"dominated by sampling noise")
    if np.isfinite(point["csi"]) and np.isfinite(lo["csi"]):
        width = hi["csi"] - lo["csi"]
        if width > 0.2:
            warns.append(f"CSI interval spans {width:.2f} -- too wide to distinguish "
                         f"models; differences under {width:.2f} are not real")

    return CIResult(point, lo, hi, n_events, N, n_boot, alpha, warns)


def attach_confidence_intervals(
    result,
    pred: np.ndarray,
    obs: np.ndarray,
    config: EvalConfig | None = None,
    mask: np.ndarray | None = None,
    n_boot: int = 1000,
    seed: int = 0,
):
    """Add CIs to an EvaluationResult, per lead time and pooled.

    Takes the same (N, L, H, W) arrays that produced the result, so the
    intervals cannot drift out of sync with the point estimates.
    """
    cfg = config or EvalConfig()
    pred = np.asarray(pred, dtype=np.float64)
    obs = np.asarray(obs)
    if cfg.obs_threshold is not None:
        obs = (obs.astype(np.float64) >= cfg.obs_threshold).astype(np.float64)
    if mask is not None:
        mask = np.broadcast_to(np.asarray(mask, dtype=bool), pred.shape)

    for li, lead in enumerate(result.per_lead):
        m = mask[:, li] if mask is not None else None
        lead.ci = bootstrap_ci(pred[:, li], obs[:, li], cfg, m,
                               n_boot=n_boot, seed=seed + li).to_dict()

    # Pooled: pass the full (N, L, H, W) array so the lead-time axis is
    # collapsed INTO each case rather than resampled as if independent.
    # Flattening to (N*L, H, W) here would treat the 30-min and 6-h frames of
    # one storm as two separate observations and shrink the interval.
    result.pooled["ci"] = bootstrap_ci(
        pred, obs, cfg, mask, n_boot=n_boot, seed=seed + 999,
    ).to_dict()
    return result


def significantly_better(a: CIResult, b: CIResult, metric: str = "csi") -> bool:
    """Is model `a` beating model `b`, or is it inside the noise?

    Deliberately conservative: requires a's lower bound to clear b's upper
    bound. Overlapping intervals are NOT evidence of a difference, and this
    is the check to run before claiming your transformer beat persistence.
    """
    lo_a, hi_b = a.lo.get(metric), b.hi.get(metric)
    if lo_a is None or hi_b is None or not (np.isfinite(lo_a) and np.isfinite(hi_b)):
        return False
    return lo_a > hi_b
