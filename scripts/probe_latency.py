"""FULL-CHAIN latency probe. Run in the first rented CUDA session.

Sub-60-second end-to-end is the entire claim against NWP, and nobody has
measured it. The forward pass is ONE TERM, not the answer:

    decode -> regrid -> normalise -> forward -> postprocess -> publish

The forward pass has been measured at 40.87 s on MPS for the full India grid
at 28M params. "4-8 s on a 4090" is a 5-10x EXTRAPOLATION, not a measurement,
and memory being O(N) does not make compute cheap -- it is still 2.4 billion
attention pairs per layer per frame. This script replaces the guess.

    python scripts/probe_latency.py --insat SCAN.h5 --era5-time 2025-08-01T23:00
"""
# nowcast_data FIRST, before torch or pyresample: it calls pin_threads(), and
# OpenMP reads those variables when its runtime LOADS. Importing torch first
# makes the pinning a no-op and the process aborts with
# "OMP: Error #15: multiple copies of the OpenMP runtime".
# This script hit exactly that on its first run.
from nowcast_data._threads import ensure_pinned_or_reexec  # noqa: E402

# Re-exec with the OpenMP variables set if they were absent. Setting them from
# Python is NOT enough for this path -- measured: this script aborts with
# "OMP: Error #15" when pin_threads() sets them in-process, and runs clean when
# the shell exports them first.
ensure_pinned_or_reexec()

import nowcast_data  # noqa: F401,E402  (thread-pinning side effect)

import argparse, json, time  # noqa: E402
from contextlib import contextmanager  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

TIMES = {}


@contextmanager
def stage(name):
    # No torch import here: this runs during CPU-side ingest too, and probing
    # the GPU before resampling is exactly what aborts the process.
    sync = globals().get("_sync")
    if sync:
        sync()
    t0 = time.perf_counter()
    yield
    if sync:
        sync()
    TIMES[name] = time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--insat", required=True, help="one L1B scan")
    ap.add_argument("--era5-time", default="", help="ISO timestamp for context")
    ap.add_argument("--dim", type=int, default=384)
    ap.add_argument("--depth", type=int, default=10)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out", default="runs/latency_probe.json")
    a = ap.parse_args()

    # ORDER MATTERS, and this is a deployment finding, not a script quirk.
    # Touching torch's GPU backend (cuda.is_available / mps.is_available)
    # initialises a second OpenMP runtime, and pyresample's pykdtree then
    # aborts the process with "OMP: Error #15". Importing torch is fine;
    # INITIALISING THE DEVICE before the resampling is not.
    #
    # So: all CPU-side ingest first, GPU second. That is also the natural
    # order for the service -- decode and regrid, then hand to the model --
    # but it has to be deliberate, and a service that probes the GPU at
    # startup to log its device would trip this on every request.
    from nowcast_data.era5 import load_fields, resample_to_grid as era5_grid
    from nowcast_data.grids import SHAPE
    from nowcast_data.insat import ingest_scan

    # --- ingest: decode + regrid, the terms most likely to dominate ---------
    with stage("insat_decode_and_regrid"):
        sat, chk = ingest_scan(a.insat, ("TIR1", "WV"), strict=False)
    print(f"  INSAT checks: {'pass' if chk.passed else 'FAIL'}")

    ctx_arr = None
    if a.era5_time:
        with stage("era5_load"):
            f = load_fields(a.era5_time)
        with stage("era5_regrid"):
            g = era5_grid(f)
        ctx_arr = np.stack([f[k] for k in sorted(f.names)])[None]

    # --- GPU only now, after all resampling is done ------------------------
    import torch
    from nowcast_model import ModelConfig, MultiTaskNowcaster

    dev = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device: {dev}")
    if dev == "cuda":
        print(f"  {torch.cuda.get_device_name(0)}")
        globals()["_sync"] = torch.cuda.synchronize
    elif dev == "mps":
        globals()["_sync"] = torch.mps.synchronize

    # --- assemble the input tensor -----------------------------------------
    with stage("assemble_tensor"):
        stack = np.stack([np.nan_to_num(sat[c]) for c in ("TIR1", "WV")])
        x = torch.from_numpy(stack[None, :, None]).float().repeat(1, 1, 2, 1, 1)
        x = torch.cat([x, torch.isfinite(
            torch.from_numpy(stack[None, :, None].astype(np.float32))
        ).float().repeat(1, 1, 2, 1, 1)], dim=1)
        x = x.to(dev)
        coords = (torch.rand(1, 48, 2, device=dev) * 2 - 1)
        ctx = (torch.from_numpy(ctx_arr).float().to(dev)
               if ctx_arr is not None else None)

    with stage("model_build"):
        m = MultiTaskNowcaster(ModelConfig(
            in_channels=x.shape[1], context_frames=2, grid_size=max(SHAPE),
            lead_steps=6, dim=a.dim, depth=a.depth,
            context_channels=(ctx.shape[1] if ctx is not None else 0))).to(dev).eval()

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

    total = sum(TIMES.values()) - TIMES.get("model_build", 0)
    print(f"\n{'stage':>28} {'sec':>8} {'% of chain':>11}")
    print("-" * 50)
    for k, v in TIMES.items():
        if k == "model_build":
            continue
        print(f"{k:>28} {v:>8.2f} {100*v/total:>10.1f}%")
    print("-" * 50)
    print(f"{'END-TO-END':>28} {total:>8.2f}")
    print(f"{'(model build, once)':>28} {TIMES.get('model_build',0):>8.2f}")

    budget = 60.0
    print(f"\n  budget {budget:.0f} s -> {'PASS' if total < budget else 'FAIL'}, "
          f"{total/budget:.0%} used")
    if TIMES["forward"] / total < 0.5:
        print(f"  the forward pass is {100*TIMES['forward']/total:.0f}% of the chain -- "
              f"optimising the model would not be the win here")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(
        {"device": dev, "stages": TIMES, "end_to_end_s": total,
         "dim": a.dim, "depth": a.depth}, indent=2))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
