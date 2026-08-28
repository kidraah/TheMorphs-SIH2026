"""The three plots worth putting in front of a judge.

Deliberately matplotlib-only and file-based -- no notebook state, no
interactivity. These regenerate identically from a saved result JSON.
"""
from __future__ import annotations

import numpy as np

from .core import EvaluationResult
from .fss import useful_scale_threshold


def _lead_labels(result: EvaluationResult) -> list[str]:
    return [f"{l.lead_minutes:g}m" if l.lead_minutes is not None else f"#{l.lead_index}"
            for l in result.per_lead]


def skill_vs_lead_time(result: EvaluationResult, path: str, models: dict | None = None):
    """The headline figure: does skill survive out to 6 hours?

    This is the plot that decides whether the project's central claim -- an
    actionable 2-6 h lead time -- is true. Baselines are overlaid because a
    curve alone is unreadable without something to beat.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.arange(len(result.per_lead))
    labels = _lead_labels(result)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    series = {"model": result}
    series.update(models or {})

    for name, res in series.items():
        axes[0].plot(x, [l.headline["csi"] for l in res.per_lead], "o-", label=name)
        axes[1].plot(x, [l.headline["pod"] for l in res.per_lead], "o-", label=f"{name} POD")
        axes[1].plot(x, [l.headline["far"] for l in res.per_lead], "s--", label=f"{name} FAR")
        axes[2].plot(x, [l.probabilistic["bss"] for l in res.per_lead], "o-", label=name)

    for ax, title in zip(axes, ("CSI", "POD (solid) / FAR (dashed)", "Brier skill score")):
        ax.set_xticks(x); ax.set_xticklabels(labels)
        ax.set_xlabel("lead time"); ax.set_title(title)
        ax.grid(alpha=0.3); ax.legend(fontsize=8)
    axes[0].set_ylim(0, 1); axes[1].set_ylim(0, 1)
    axes[2].axhline(0, color="k", lw=1, ls=":")  # climatology = zero skill

    fig.suptitle(f"skill vs lead time  (p>={result.config['headline_threshold']}, "
                 f"base rate {result.pooled['base_rate']:.2e})")
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def fss_by_scale(result: EvaluationResult, path: str):
    """At what spatial scale does the forecast become useful?

    A model can be worthless at 4 km and genuinely useful at 50 km. For
    disaster management that distinction is the whole product decision, so
    it gets its own figure rather than a footnote.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = result.config["headline_threshold"]
    kms = list(result.config["neighborhood_km"])
    fig, ax = plt.subplots(figsize=(7, 4.5))

    for l, lab in zip(result.per_lead, _lead_labels(result)):
        ys = [next((f["fss"] for f in l.fss
                    if f["neighborhood_km"] == km and f["threshold"] == t), np.nan)
              for km in kms]
        ax.plot(kms, ys, "o-", label=lab)

    ax.axhline(useful_scale_threshold(result.pooled["base_rate"]),
               color="k", ls="--", lw=1, label="'useful' (0.5 + f0/2)")
    ax.set_xlabel("neighbourhood width (km)"); ax.set_ylabel("FSS")
    ax.set_ylim(0, 1); ax.grid(alpha=0.3); ax.legend(fontsize=8, title="lead")
    ax.set_title("FSS by spatial scale")
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def reliability_diagram(result: EvaluationResult, path: str):
    """Does a stated 70% mean 70%?

    An operator who evacuates on your probabilities needs this to sit on the
    diagonal. A model far below it is confidently wrong, which is worse for
    trust than being uncertain.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rel = result.pooled["probabilistic"]["reliability"]
    mf = np.array(rel["mean_forecast"], dtype=float)
    of = np.array(rel["observed_freq"], dtype=float)
    counts = np.array(rel["counts"], dtype=float)

    fig, (ax, axh) = plt.subplots(2, 1, figsize=(6, 7),
                                  gridspec_kw={"height_ratios": [3, 1]}, sharex=True)
    ax.plot([0, 1], [0, 1], "k:", lw=1, label="perfect")
    ax.axhline(result.pooled["base_rate"], color="gray", lw=1, ls="--", label="climatology")
    ok = np.isfinite(mf) & np.isfinite(of)
    ax.plot(mf[ok], of[ok], "o-", label="forecast")
    ax.set_ylabel("observed frequency"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    ax.set_title(f"reliability  (BSS {result.pooled['probabilistic']['bss']:.3f})")

    edges = np.array(rel["bin_edges"], dtype=float)
    axh.bar(edges[:-1], np.maximum(counts, 1), width=np.diff(edges),
            align="edge", edgecolor="k", linewidth=0.4)
    axh.set_yscale("log")  # rare events -> counts span orders of magnitude
    axh.set_xlabel("forecast probability"); axh.set_ylabel("count"); axh.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)
