"""FULL-CHAIN latency probe, measured through the deployed service shape.

Sub-60-second end-to-end is the entire claim against NWP. The forward pass
is ONE TERM, not the answer:

    decode -> regrid -> normalise -> forward -> postprocess -> publish

WHY THE TWO STAGES ARE REPORTED SEPARATELY
------------------------------------------
They scale differently, so their ratio at the small config says nothing
about the ratio at the real one:

    ingest (decode + regrid)   INVARIANT in model size. Fixed work per scan:
                               HDF5 read, LUT decode, one KD-tree query.
    forward                    scales with dim, depth and token count.

Measured here at dim 128 on MPS, ingest was 40% of a 6.70 s chain. At dim
384 / depth 10 the forward term grows and ingest does not, so the shares
must be re-read at the real config rather than carried over. If ingest still
dominates at 28M, model optimisation is the wrong place to spend effort; if
forward dominates, the tiling and attention-range questions become the
budget. The script prints the split and says which way it landed.

This probe runs ingest in the exec'd worker (nowcast_serve), i.e. the same
shape the service deploys in -- an earlier version measured an in-process
path that the service cannot legally use.

    python scripts/probe_latency.py --insat SCAN.h5 --era5-time 2025-08-01T23:00
"""
from __future__ import annotations

import argparse
import json
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

TIMES = {}


@contextmanager
def stage(name):
    sync = globals().get("_sync")
    if sync:
        sync()
    t0 = time.perf_counter()
    yield
    if sync:
        sync()
    TIMES[name] = time.perf_counter() - t0


# Stages that do not move when the model grows.
MODEL_INVARIANT = ("insat_decode_and_regrid", "era5_load_and_regrid",
                   "worker_start")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--insat", required=True, help="one L1B scan")
    ap.add_argument("--era5-time", default="", help="ISO timestamp for context")
    ap.add_argument("--dim", type=int, default=384)
    ap.add_argument("--depth", type=int, default=10)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--grid", type=int, default=0,
                    help="model grid side; 0 = full India grid")
    ap.add_argument("--out", default="runs/latency_probe.json")
    a = ap.parse_args()

    # No import-order dance and no re-exec: ingest happens in a separate
    # process that has never imported torch, so the GPU may be touched
    # whenever it is convenient. See docs/SERVICE_ARCHITECTURE.md.
    from nowcast_serve import IngestWorker

    with stage("worker_start"):
        worker = IngestWorker().start()
    print(f"ingest worker pid {worker.child_pid} "
          f"({TIMES['worker_start']:.2f}s, once at boot)")

    with stage("insat_decode_and_regrid"):
        sat, check = worker.ingest(a.insat, ("TIR1", "WV"))
    print(f"  INSAT checks: {'pass' if check['passed'] else 'FAIL'}")
    if not check["passed"]:
        print("  " + "\n  ".join(check["failures"]))

    ctx_arr = None
    if a.era5_time:
        with stage("era5_load_and_regrid"):
            fields, names = worker.era5(a.era5_time)
        ctx_arr = np.stack([fields[k] for k in names])[None]

    import torch
    from nowcast_model import ModelConfig, MultiTaskNowcaster
    from nowcast_data.grids import SHAPE

    dev = ("cuda" if torch.cuda.is_available()
           else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device: {dev}")
    if dev == "cuda":
        print(f"  {torch.cuda.get_device_name(0)}")
        globals()["_sync"] = torch.cuda.synchronize
    elif dev == "mps":
        globals()["_sync"] = torch.mps.synchronize

    grid = a.grid or max(SHAPE)

    with stage("assemble_tensor"):
        stack = np.stack([np.nan_to_num(sat[c]) for c in ("TIR1", "WV")])
        finite = np.isfinite(np.stack([sat[c] for c in ("TIR1", "WV")]))
        both = np.concatenate([stack, finite.astype(np.float32)], 0)[:, :grid, :grid]
        x = torch.from_numpy(both[None, :, None]).float().repeat(1, 1, 2, 1, 1).to(dev)
        coords = (torch.rand(1, 48, 2, device=dev) * 2 - 1)
        ctx = (torch.from_numpy(ctx_arr).float().to(dev)
               if ctx_arr is not None else None)

    with stage("model_build"):
        m = MultiTaskNowcaster(ModelConfig(
            in_channels=x.shape[1], context_frames=2, grid_size=grid,
            lead_steps=6, dim=a.dim, depth=a.depth,
            context_channels=(ctx.shape[1] if ctx is not None else 0))).to(dev).eval()
    n_par = sum(p.numel() for p in m.parameters())

    with torch.no_grad():
        m(x, coords, context=ctx)                       # warmup
        fwd = []
        for _ in range(a.repeats):
            with stage("forward"):
                out = m(x, coords, context=ctx)
            fwd.append(TIMES["forward"])
    TIMES["forward"] = float(np.median(fwd))

    with stage("postprocess"):
        probs = {k: torch.sigmoid(v).float().cpu().numpy() for k, v in out.items()}
        _ = {k: (v >= 0.3).sum() for k, v in probs.items()}

    worker.stop()

    # -- report -------------------------------------------------------------
    per_request = {k: v for k, v in TIMES.items()
                   if k not in ("model_build", "worker_start")}
    total = sum(per_request.values())
    print(f"\nconfig: dim {a.dim} depth {a.depth} grid {grid} "
          f"-> {n_par/1e6:.1f}M params")
    print(f"{'stage':>28} {'sec':>8} {'% of chain':>11}")
    print("-" * 50)
    for k, v in per_request.items():
        print(f"{k:>28} {v:>8.2f} {100*v/total:>10.1f}%")
    print("-" * 50)
    print(f"{'END-TO-END (per request)':>28} {total:>8.2f}")
    print(f"{'model build (once)':>28} {TIMES.get('model_build', 0):>8.2f}")
    print(f"{'worker start (once)':>28} {TIMES.get('worker_start', 0):>8.2f}")

    # The split the ratio question turns on.
    ingest = sum(v for k, v in per_request.items() if k in MODEL_INVARIANT)
    forward = TIMES["forward"]
    print(f"\n  ingest  (model-INVARIANT) {ingest:>7.2f} s  {100*ingest/total:>5.1f}%")
    print(f"  forward (scales with size) {forward:>6.2f} s  {100*forward/total:>5.1f}%")
    if forward > ingest:
        print(f"  -> the model dominates at this config ({forward/ingest:.1f}x ingest). "
              f"Tiling and attention range are the budget.")
    else:
        print(f"  -> ingest still dominates at this config ({ingest/forward:.1f}x forward). "
              f"Optimising the model would not be the win; decode/regrid is.")

    budget = 60.0
    print(f"\n  budget {budget:.0f} s -> {'PASS' if total < budget else 'FAIL'}, "
          f"{total/budget:.0%} used")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(
        {"device": dev, "stages": TIMES, "end_to_end_s": total,
         "ingest_s": ingest, "forward_s": forward, "params": n_par,
         "dim": a.dim, "depth": a.depth, "grid": grid}, indent=2))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
