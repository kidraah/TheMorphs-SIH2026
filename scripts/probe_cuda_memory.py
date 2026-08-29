"""Run this FIRST on the rented CUDA box, before any training.

Answers the question MPS could not: does a full-grid India forward pass
(864 x 912 = 787,968 cells, 86x a SEVIR tile) fit in 24 GB, or does
inference need tiling with overlap?

That determines the end-to-end latency number, which is the headline claim
against NWP -- tiled inference with overlap costs several forward passes plus
seam handling, and that has to be measured, not assumed.

    python scripts/probe_cuda_memory.py --dim 320 --depth 8
"""
import argparse, json, time
from pathlib import Path

import torch

from nowcast_data.grids import SHAPE
from nowcast_model import ModelConfig, MultiTaskNowcaster

ap = argparse.ArgumentParser()
ap.add_argument("--dim", type=int, default=320)
ap.add_argument("--depth", type=int, default=8)
ap.add_argument("--in-channels", type=int, default=4)
ap.add_argument("--out", default="runs/cuda_memory_probe.json")
a = ap.parse_args()

if not torch.cuda.is_available():
    raise SystemExit("no CUDA device -- this probe is for the rented box")

dev = torch.device("cuda")
name = torch.cuda.get_device_name(0)
total = torch.cuda.get_device_properties(0).total_memory / 2**30
print(f"device: {name}  ({total:.1f} GiB)\n")


def probe(grid, batch, train: bool):
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    m = MultiTaskNowcaster(ModelConfig(in_channels=a.in_channels, context_frames=2,
          grid_size=grid, lead_steps=6, dim=a.dim, depth=a.depth)).to(dev)
    x = torch.randn(batch, a.in_channels, 2, grid, grid, device=dev)
    co = (torch.rand(batch, 48, 2, device=dev) * 2 - 1)
    torch.cuda.synchronize(); t0 = time.time()
    if train:
        opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
        out = m(x, co); sum(v.float().sum() for v in out.values()).backward()
        opt.step()
    else:
        m.eval()
        with torch.no_grad():
            m(x, co)
    torch.cuda.synchronize()
    dt = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 2**30
    del m, x, co
    torch.cuda.empty_cache()
    return peak, dt


rows = []
print(f"{'mode':>9} {'grid':>10} {'batch':>6} {'peak GiB':>9} {'sec':>7}  fits?")
plan = [("train", 96, b) for b in (8, 16, 32, 64)] + \
       [("infer", g, 1) for g in (96, 192, 384, 576, 768, max(SHAPE))]
for mode, grid, batch in plan:
    try:
        peak, dt = probe(grid, batch, train=(mode == "train"))
        fits = "yes" if peak < total * 0.9 else "TIGHT"
        print(f"{mode:>9} {grid:>10} {batch:>6} {peak:>9.2f} {dt:>7.2f}  {fits}")
        rows.append(dict(mode=mode, grid=grid, batch=batch, peak_gib=peak, sec=dt))
    except torch.cuda.OutOfMemoryError:
        print(f"{mode:>9} {grid:>10} {batch:>6} {'OOM':>9} {'--':>7}  NO -> tiling needed")
        rows.append(dict(mode=mode, grid=grid, batch=batch, peak_gib=None, oom=True))
        torch.cuda.empty_cache()

print(f"\nIndia grid is {SHAPE[1]} x {SHAPE[0]}; the largest square probed above "
      f"is {max(SHAPE)}.")
print("If full-grid inference OOMs, tiling with overlap is required and the "
      "latency claim must be measured on the TILED path, not a single pass.")
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
Path(a.out).write_text(json.dumps(
    {"device": name, "total_gib": total, "dim": a.dim, "depth": a.depth,
     "rows": rows}, indent=2))
print(f"wrote {a.out}")
