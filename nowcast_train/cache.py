"""Pre-decoded cache. The single highest-value performance change available.

Measured on the real store: 1.40 steps/s against 5.17 steps/s for CPU-only
compute, so ~70% of every step was I/O -- reading ~14 MB of HDF5 per sample
(three files, random access) and re-doing the decode, temporal subsample and
resolution matching every epoch, identically, forever.

The cache does that work once. Each event becomes a small array already
decoded, already at the analysis grid, already at the scan cadence:

    3 channels x 9 frames x 96 x 96 float16  ~= 500 KB/event
    12,098 events                            ~= 6 GB   (vs 229 GB raw)

Re-runnable and parameterised
-----------------------------
Every resolution decision on this project has been revised at least once --
the WV 8 km treatment, the VIL 12 km labels, the 4 km grid. So nothing here
is hardcoded: resolution, crop, channel set and temporal window are all
config, and a manifest records exactly which config produced a given cache.

`CachedDataset` REFUSES to load a cache whose manifest does not match the
config it was asked for. Silently training on a stale cache -- built before a
resolution decision changed -- would be undetectable in the loss and wrong in
every downstream number.

The full temporal extent is cached, not a pre-cut window, so context/horizon
lengths can change without rebuilding.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class CacheConfig:
    """Everything that changes the bytes on disk. Hashed into the cache id."""
    grid_size: int = 96                 # cells per side at target_km
    target_km: float = 4.0
    cadence_min: float = 30.0
    input_channels: tuple = ("ir107", "ir069")
    input_sensor_km: tuple = (4.0, 8.0)   # INSAT counterparts
    target_channel: str = "vil"
    label_km: float = 12.0
    dtype: str = "float16"
    crop: tuple | None = None           # (row0, col0, h, w) or None for full
    version: int = 1                    # bump to invalidate every cache

    def fingerprint(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, default=str)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def to_dict(self) -> dict:
        return asdict(self) | {"fingerprint": self.fingerprint()}


class CacheMismatch(RuntimeError):
    """Raised rather than silently training on a stale cache."""


def _manifest_path(root) -> Path:
    return Path(root) / "manifest.json"


def write_manifest(root, config: CacheConfig, event_ids: Sequence[str],
                   days: Sequence[str], source: str, extra: dict | None = None):
    m = {
        "config": config.to_dict(),
        "n_events": len(event_ids),
        "event_ids": list(event_ids),
        "days": [str(d) for d in days],
        "source": str(source),
        "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "extra": extra or {},
    }
    Path(root).mkdir(parents=True, exist_ok=True)
    _manifest_path(root).write_text(json.dumps(m, indent=2))
    return m


def read_manifest(root) -> dict:
    p = _manifest_path(root)
    if not p.exists():
        raise CacheMismatch(f"no manifest at {p} -- refusing to use an "
                            f"unidentified cache. Rebuild with build_cache().")
    return json.loads(p.read_text())


def verify_manifest(root, config: CacheConfig) -> dict:
    """Refuse a cache built under a different config."""
    m = read_manifest(root)
    got = m["config"].get("fingerprint")
    want = config.fingerprint()
    if got != want:
        diffs = [f"    {k}: cache={m['config'].get(k)!r} requested={v!r}"
                 for k, v in config.to_dict().items()
                 if k != "fingerprint" and m["config"].get(k) != v]
        raise CacheMismatch(
            f"cache at {root} was built with a different config "
            f"({got} != {want}).\n" + "\n".join(diffs) +
            f"\nRebuild it, or point at the cache that matches. Training on a "
            f"stale cache is undetectable in the loss.")
    return m


def build_cache(loader, event_ids: Sequence[str], out_dir,
                config: CacheConfig | None = None, overwrite: bool = False,
                log_every: int = 500) -> dict:
    """Decode, resample and store every event once. Resumable."""
    from nowcast_data.grid import match_sensor_resolution, subsample_time
    from nowcast_data.sevir import SEVIR_CADENCE_MIN

    cfg = config or CacheConfig()
    root = Path(out_dir) / cfg.fingerprint()
    root.mkdir(parents=True, exist_ok=True)
    specs = {s.img_type: s for s in list(loader.cfg.inputs) + [loader.cfg.target]}

    done, skipped = [], 0
    t0 = time.time()
    for i, eid in enumerate(event_ids):
        dest = root / f"{eid}.npz"
        if dest.exists() and not overwrite:
            skipped += 1
            done.append(eid)
            continue
        try:
            chans = []
            for name, sensor_km in zip(cfg.input_channels, cfg.input_sensor_km):
                spec = specs[name]
                raw = loader._read_raw(eid, spec)
                timed = subsample_time(raw, SEVIR_CADENCE_MIN, cfg.cadence_min, axis=0)
                chans.append(match_sensor_resolution(timed, spec.native_km,
                                                     sensor_km, cfg.target_km))
            tspec = specs[cfg.target_channel]
            traw = subsample_time(loader._read_raw(eid, tspec),
                                  SEVIR_CADENCE_MIN, cfg.cadence_min, axis=0)
            y = match_sensor_resolution(traw, tspec.native_km, cfg.label_km,
                                        cfg.target_km)
            x = np.stack(chans)                       # (C, T, H, W)
            if cfg.crop:
                r, c, h, w = cfg.crop
                x, y = x[..., r:r + h, c:c + w], y[..., r:r + h, c:c + w]
            np.savez(dest, x=x.astype(cfg.dtype), y=y.astype(cfg.dtype))
            done.append(eid)
        except Exception as e:                        # one bad event must not
            print(f"  SKIP {eid}: {type(e).__name__}: {e}")   # kill the build
        if log_every and i and i % log_every == 0:
            rate = (i + 1) / (time.time() - t0)
            print(f"  {i + 1}/{len(event_ids)} events  {rate:.1f}/s  "
                  f"eta {(len(event_ids) - i) / max(rate, 1e-9) / 60:.0f} min")

    days = [loader.catalog.day_of(e) for e in done]
    m = write_manifest(root, cfg, done, days, source=str(loader.cfg.data_root),
                       extra={"reused_existing": skipped,
                              "build_seconds": round(time.time() - t0, 1)})
    print(f"cache ready: {root}  ({len(done)} events, {skipped} reused, "
          f"{sum(f.stat().st_size for f in root.glob('*.npz')) / 2**30:.2f} GB)")
    return m


class CachedEvents:
    """Random access to a verified cache. Replaces SEVIRLoader in training."""

    def __init__(self, root, config: CacheConfig | None = None):
        self.root = Path(root)
        self.cfg = config or CacheConfig()
        self.manifest = verify_manifest(self.root, self.cfg)
        self.event_ids = list(self.manifest["event_ids"])
        self._days = dict(zip(self.event_ids, self.manifest["days"]))

    def __len__(self) -> int:
        return len(self.event_ids)

    def day_of(self, eid: str) -> str:
        return self._days[eid]

    def load(self, eid: str) -> tuple:
        with np.load(self.root / f"{eid}.npz") as d:
            return d["x"].astype(np.float32), d["y"].astype(np.float32)
