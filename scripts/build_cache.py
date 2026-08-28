"""Build the pre-decoded SEVIR cache. Re-runnable; skips what exists.

    .venv/bin/python scripts/build_cache.py
    .venv/bin/python scripts/build_cache.py --label-km 8 --out data/cache

Every parameter that changes the bytes is in CacheConfig and hashed into the
cache directory name, so two configs cannot collide and a stale cache cannot
be loaded silently.
"""
import argparse, warnings
from pathlib import Path

warnings.filterwarnings("ignore")

from nowcast_data.sevir import DEFAULT_CHANNELS, VIL_CHANNEL, SEVIRConfig, SEVIRLoader
from nowcast_train.cache import CacheConfig, build_cache

ap = argparse.ArgumentParser()
ap.add_argument("--store", default="data/sevir")
ap.add_argument("--out", default="data/cache")
ap.add_argument("--label-km", type=float, default=12.0)
ap.add_argument("--cadence-min", type=float, default=30.0)
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--overwrite", action="store_true")
a = ap.parse_args()

loader = SEVIRLoader(SEVIRConfig.from_store(
    Path(a.store), inputs=[DEFAULT_CHANNELS["ir107"], DEFAULT_CHANNELS["ir069"]],
    target=VIL_CHANNEL, context_frames=2, horizon_frames=6))

cfg = CacheConfig(grid_size=loader.cfg.grid_size, target_km=loader.cfg.target_km,
                  cadence_min=a.cadence_min, label_km=a.label_km)
ids = loader.event_ids()
if a.limit:
    ids = ids[:a.limit]
print(f"caching {len(ids)} events -> {Path(a.out) / cfg.fingerprint()}")
print(f"config: {cfg.to_dict()}")
build_cache(loader, ids, a.out, cfg, overwrite=a.overwrite)
