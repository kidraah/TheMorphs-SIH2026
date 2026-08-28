"""Resampling between sensor resolutions.

The one idea in this file
-------------------------
Matching a training dataset to an operational sensor is TWO steps, not one:

    1. degrade to what the operational sensor can actually resolve
    2. resample that onto the common analysis grid

Collapsing them into a single resize is the mistake. INSAT water vapour is
8 km native but will live on a 4 km analysis grid; if SEVIR's 2 km water
vapour is resized straight to 4 km, the model trains on moisture gradients
sharper than INSAT can ever deliver and then meets blurrier data at
inference. It will have learned to depend on detail that does not exist
operationally -- and IWV variation is the cornerstone of this project's
method, so that is the worst channel to get wrong.

Going 2 km -> 8 km -> 4 km looks wasteful and is the entire point: the
result lives on a 4 km grid while carrying only 8 km of real information,
which is exactly the operational situation.
"""
from __future__ import annotations

import numpy as np


def block_mean(arr: np.ndarray, factor: int) -> np.ndarray:
    """Area-average downsample by an integer factor over the last two axes.

    Area-averaging (not subsampling, not bilinear) approximates what a
    coarser sensor does: it integrates radiance over a larger footprint.
    Point-subsampling would keep sharp features a real sensor would smear.
    """
    if factor < 1:
        raise ValueError(f"factor must be >= 1, got {factor}")
    if factor == 1:
        return arr.astype(np.float64, copy=True)

    h, w = arr.shape[-2:]
    if h % factor or w % factor:
        raise ValueError(
            f"{h}x{w} is not divisible by {factor}; crop or pad first rather "
            f"than letting an uneven block silently weight edge pixels differently")

    lead = arr.shape[:-2]
    out = arr.astype(np.float64).reshape(*lead, h // factor, factor, w // factor, factor)
    return out.mean(axis=(-3, -1))


def nearest_upsample(arr: np.ndarray, factor: int) -> np.ndarray:
    """Replicate pixels by an integer factor over the last two axes.

    Nearest, not bilinear, and deliberately: upsampling creates no
    information, and a blocky field is an honest representation of that.
    Bilinear would produce smooth gradients that look like real structure
    and are not.
    """
    if factor < 1:
        raise ValueError(f"factor must be >= 1, got {factor}")
    if factor == 1:
        return arr.astype(np.float64, copy=True)
    return np.repeat(np.repeat(arr.astype(np.float64), factor, axis=-2), factor, axis=-1)


def _ratio(a: float, b: float) -> int:
    """b / a as an exact integer, or raise."""
    r = b / a
    n = int(round(r))
    if n < 1 or abs(r - n) > 1e-9:
        raise ValueError(f"{b} / {a} = {r} is not an integer ratio; "
                         f"non-integer resampling needs an explicit reprojection")
    return n


def match_sensor_resolution(
    arr: np.ndarray,
    native_km: float,
    sensor_km: float,
    target_km: float,
) -> np.ndarray:
    """Degrade to `sensor_km`, then place on the `target_km` grid.

    arr          : (..., H, W) at `native_km` resolution (SEVIR)
    native_km    : the training data's true resolution (SEVIR: 2 km IR, 1 km VIL)
    sensor_km    : what the OPERATIONAL sensor resolves (INSAT: 4 km TIR, 8 km WV)
    target_km    : the common analysis grid (4 km)

    With sensor_km == target_km this is a plain downsample. With
    sensor_km > target_km -- the INSAT water vapour case -- the field is
    degraded to 8 km and then blocked back up to 4 km, so it occupies the
    analysis grid while carrying only the information INSAT really has.
    """
    if sensor_km < native_km:
        raise ValueError(
            f"sensor_km ({sensor_km}) finer than native_km ({native_km}): the "
            f"training data cannot supply detail the source does not have")

    degraded = block_mean(arr, _ratio(native_km, sensor_km))
    if sensor_km == target_km:
        return degraded
    if sensor_km > target_km:
        return nearest_upsample(degraded, _ratio(target_km, sensor_km))
    # sensor finer than the analysis grid: average the rest of the way down
    return block_mean(degraded, _ratio(sensor_km, target_km))


def subsample_time(arr: np.ndarray, native_min: float, target_min: float,
                   axis: int = 0) -> np.ndarray:
    """Take every Nth frame to match the operational scan cadence.

    The temporal twin of the spatial argument above. SEVIR is 5-minute
    frames; INSAT-3D full disk is 30 minutes. A backbone trained on 5-minute
    motion learns storm evolution at a timescale the operational feed never
    shows it.
    """
    step = _ratio(native_min, target_min)
    return np.take(arr, np.arange(0, arr.shape[axis], step), axis=axis)
