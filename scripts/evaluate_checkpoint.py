"""Full three-head scorecard with per-head operating points and calibration.

Selection happens on VALIDATION, reporting on TEST. Fitting the threshold and
the calibrator on the same data they are scored against would be optimistic
by construction -- the thresholds are chosen to maximise the very metric then
reported.
"""
import argparse, json, warnings
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore")

from nowcast_eval import EvalConfig, attach_confidence_intervals, evaluate_multi
from nowcast_model import ModelConfig, MultiTaskNowcaster
from nowcast_train import (CacheConfig, CachedEvents, CachedSEVIRDataset,
                           DatasetConfig, IsotonicCalibrator, collate,
                           compute_channel_stats, episode_ids, load_checkpoint,
                           select_threshold, split_by_time)
from nowcast_data.sevir import DEFAULT_CHANNELS, VIL_CHANNEL, SEVIRConfig, SEVIRLoader

ap = argparse.ArgumentParser()
ap.add_argument("--checkpoint", default="runs/cached/last.pt")
ap.add_argument("--cache", default="data/cache/f4e481f79e2de287")
ap.add_argument("--store", default="data/sevir")
ap.add_argument("--dim", type=int, default=128)
ap.add_argument("--depth", type=int, default=4)
ap.add_argument("--batch-size", type=int, default=16)
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--device", default="cpu")
ap.add_argument("--n-boot", type=int, default=300)
ap.add_argument("--out", default="runs/cached/scorecard.json")
a = ap.parse_args()

cache = CachedEvents(a.cache, CacheConfig())
rep = split_by_time(cache.event_ids, [cache.day_of(e) for e in cache.event_ids])
loader = SEVIRLoader(SEVIRConfig.from_store(
    Path(a.store), inputs=[DEFAULT_CHANNELS["ir107"], DEFAULT_CHANNELS["ir069"]],
    target=VIL_CHANNEL, context_frames=2, horizon_frames=6))
stats = compute_channel_stats(loader, rep["train"].event_ids, n_sample=100)
dcfg = DatasetConfig(stats=stats)

def infer(ids):
    ds = CachedSEVIRDataset(cache, ids, dcfg)
    m = MultiTaskNowcaster(ModelConfig(in_channels=ds.in_channels, context_frames=2,
          grid_size=cache.cfg.grid_size, lead_steps=6, dim=a.dim, depth=a.depth))
    load_checkpoint(a.checkpoint, model=m)
    m.to(a.device).eval()
    P, T = {}, {}
    with torch.no_grad():
        for x, y, mk, co, _ in DataLoader(ds, batch_size=a.batch_size,
                                          collate_fn=collate):
            for k, v in m(x.to(a.device), co.to(a.device)).items():
                P.setdefault(k, []).append(torch.sigmoid(v).float().cpu().numpy())
                T.setdefault(k, []).append(y[k].numpy())
    return ({k: np.concatenate(v) for k, v in P.items()},
            {k: np.concatenate(v) for k, v in T.items()})

val_ids = rep["val"].event_ids[:a.limit or None]
test_ids = rep["test"].event_ids[:a.limit or None]
print(f"val {len(val_ids)} events ({rep['val'].n_episodes} episodes)  ->  "
      f"test {len(test_ids)} events ({rep['test'].n_episodes} episodes)\n")

print("inferring on val (for threshold + calibration selection)...", flush=True)
Pv, Tv = infer(val_ids)
print("inferring on test (for reporting)...", flush=True)
Pt, Tt = infer(test_ids)

heads = list(Pv)
print("\n=== selected on VALIDATION ===")
cals, thrs = {}, {}
for k in heads:
    cals[k] = IsotonicCalibrator.fit(Pv[k], Tv[k])
    pv_cal = cals[k].transform(Pv[k])
    thrs[k], sc = select_threshold(pv_cal, Tv[k], metric="sedi")
    print(f"  {k:14s} raw max {Pv[k].max():.3f} -> calibrated max {pv_cal.max():.5f}"
          f"   threshold {thrs[k]:.5f}  (val SEDI {sc:.3f})")

print("\n=== applied to TEST ===")
Pt_cal = {k: cals[k].transform(Pt[k]) for k in heads}
cfgs = {k: EvalConfig(name=k, grid_km=4.0, headline_threshold=float(thrs[k]),
                      geometry=("point" if k == "cloudburst" else "grid"),
                      thresholds=tuple(sorted({0.05, 0.1, 0.2, float(thrs[k])})))
        for k in heads}
res = evaluate_multi(Pt_cal, Tt, cfgs, lead_minutes=[30, 60, 90, 120, 150, 180])
groups = episode_ids([cache.day_of(e) for e in test_ids])
for k in heads:
    attach_confidence_intervals(res[k], Pt_cal[k], Tt[k], cfgs[k],
                                n_boot=a.n_boot, groups=groups)
print(res.summary_table().split("--- ")[0])

Path(a.out).parent.mkdir(parents=True, exist_ok=True)
Path(a.out).write_text(json.dumps(
    {"thresholds": {k: float(v) for k, v in thrs.items()},
     "scorecard": {k: res[k].pooled["headline"] | {
         "sedi_ci": res[k].pooled["ci"]["lo"].get("sedi"),
         "sedi_hi": res[k].pooled["ci"]["hi"].get("sedi")} for k in heads}},
    indent=2, default=float))
print(f"\nwrote {a.out}")
