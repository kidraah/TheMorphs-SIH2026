"""A miniature SEVIR-shaped store, for testing the loader without the download.

Mirrors the real layout: a CATALOG.csv indexing HDF5 files that hold
(events, H, W, frames) arrays keyed by image type. Small enough to build in
a test, structured enough that code which works here works on the real
thing -- with the caveat that it cannot verify the real VIL encoding or the
real column spellings, which need checking against actual data.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .sevir import SEVIR_FRAMES

# SEVIR image sizes over the same 384 km domain
SIZES = {"vis": 768, "ir069": 192, "ir107": 192, "vil": 384}
SCALES = {"vis": 1e-4, "ir069": 1e-2, "ir107": 1e-2, "vil": 1.0}


def _storm(size: int, frames: int, rng, drift: float) -> np.ndarray:
    """A blob that drifts, so downsampling and time-subsampling are visible."""
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64)
    out = np.zeros((size, size, frames))
    y0, x0 = size * 0.4, size * 0.3
    r = size * 0.08
    for t in range(frames):
        d2 = (yy - (y0 + drift * t)) ** 2 + (xx - (x0 + drift * t)) ** 2
        out[..., t] = np.exp(-d2 / (2 * r ** 2))
    return out + rng.normal(0, 0.01, out.shape)


def build_store(root: Path, n_events: int = 6, n_days: int = 2,
                img_types=("ir069", "ir107", "vil"), seed: int = 0) -> Path:
    """Write a small store and return its root. Returns path containing
    CATALOG.csv and data/."""
    import h5py
    import pandas as pd

    root = Path(root)
    (root / "data").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    rows = []
    for img_type in img_types:
        size = SIZES[img_type]
        fname = f"data/SEVIR_{img_type.upper()}_TEST.h5"
        arr = np.zeros((n_events, size, size, SEVIR_FRAMES), dtype=np.float32)
        for i in range(n_events):
            arr[i] = _storm(size, SEVIR_FRAMES, rng, drift=size / 400.0)
        # store in raw units so the loader's `scale` has something to undo
        raw = (arr / SCALES[img_type]).astype(np.int16 if img_type != "vil" else np.uint8)
        with h5py.File(root / fname, "w") as fh:
            fh.create_dataset(img_type, data=raw)
            fh.create_dataset("id", data=np.array(
                [f"E{i:03d}".encode() for i in range(n_events)]))

        for i in range(n_events):
            # events cluster onto a few storm days -- the whole reason the
            # bootstrap needs day blocking
            day = f"2019-06-{1 + (i % n_days):02d}"
            rows.append({
                "id": f"E{i:03d}",
                "file_name": fname,
                "file_index": i,
                "img_type": img_type,
                "time_utc": f"{day} 12:00:00",
                "event_type": "Thunderstorm Wind",
                "pct_missing": 0.0,
            })

    pd.DataFrame(rows).to_csv(root / "CATALOG.csv", index=False)
    return root
