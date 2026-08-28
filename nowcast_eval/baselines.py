"""Baselines. The bar the transformer has to clear.

A deep model with CSI 0.31 sounds like a result until you learn that
simply assuming the sky stays as it is scores 0.34. Report every model
against these, always, or the number means nothing.
"""
from __future__ import annotations

import numpy as np


def persistence(last_obs: np.ndarray, n_lead: int) -> np.ndarray:
    """Eulerian persistence: whatever is happening now keeps happening, in place.

    The dumbest possible forecast, and at 30-60 min lead times it is
    genuinely hard to beat. If your model loses to this at short lead,
    something is wrong with the model, not with the baseline.

    last_obs : (N, H, W) most recent observed field (binary or probability)
    returns  : (N, n_lead, H, W)
    """
    last = np.asarray(last_obs, dtype=np.float64)
    if last.ndim != 3:
        raise ValueError(f"expected (N, H, W), got {last.shape}")
    return np.repeat(last[:, None], n_lead, axis=1)


def climatology(base_rate: float, shape: tuple[int, ...]) -> np.ndarray:
    """Predict the long-run event frequency everywhere, always.

    This is the reference the Brier skill score uses. A model with BSS <= 0
    has learned nothing this baseline did not already know.
    """
    if not 0.0 <= base_rate <= 1.0:
        raise ValueError("base_rate must be in [0, 1]")
    return np.full(shape, float(base_rate), dtype=np.float64)


def pysteps_extrapolation(
    frames: np.ndarray,
    n_lead: int,
    threshold: float | None = None,
) -> np.ndarray:
    """Lagrangian persistence via pySTEPS optical flow.

    The serious baseline: estimate the motion field from recent frames and
    advect the current echo forward. This is what operational nowcasting
    actually does, and beating it at 2-6 hours -- where storms initiate and
    decay rather than merely translate -- is the real claim of the project.

    Kept behind a lazy import and an adapter so an optional dependency can
    never break the harness core.

    frames : (N, T, H, W) recent observed sequence, oldest first
    """
    try:
        from pysteps import motion, nowcasts
    except ImportError as e:  # pragma: no cover - optional dependency
        raise ImportError(
            "pySTEPS is not installed. `uv pip install pysteps` to enable this "
            "baseline, or omit it -- the harness core does not need it."
        ) from e

    frames = np.asarray(frames, dtype=np.float64)
    if frames.ndim != 4:
        raise ValueError(f"expected (N, T, H, W), got {frames.shape}")

    oflow = motion.get_method("lucaskanade")
    extrap = nowcasts.get_method("extrapolation")

    out = np.empty((frames.shape[0], n_lead, *frames.shape[2:]), dtype=np.float64)
    for i, seq in enumerate(frames):
        v = oflow(seq)
        fc = extrap(seq[-1], v, n_lead)
        out[i] = np.nan_to_num(fc, nan=0.0)

    if threshold is not None:
        out = (out >= threshold).astype(np.float64)
    return np.clip(out, 0.0, 1.0)
