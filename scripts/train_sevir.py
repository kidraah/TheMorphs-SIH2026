"""Pretrain the multi-task nowcaster on SEVIR.

    .venv/bin/python scripts/train_sevir.py --smoke      # tiny, minutes
    .venv/bin/python scripts/train_sevir.py              # full run

Resumable: rerun the same --run-dir after a spot reclaim and it continues
from the last epoch with optimizer state intact.
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

from nowcast_data.sevir import DEFAULT_CHANNELS, VIL_CHANNEL, SEVIRConfig, SEVIRLoader
from nowcast_model import ModelConfig, MultiTaskNowcaster
from nowcast_train import (DatasetConfig, NaNPolicy, SEVIRDataset, TargetConfig,
                           TrainConfig, compute_channel_stats, episode_ids,
                           split_sevir)

warnings.filterwarnings("ignore", category=UserWarning)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default="data/sevir")
    ap.add_argument("--run-dir", default="runs/sevir-pretrain")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--nan-policy", default="mask",
                    choices=[p.value for p in NaNPolicy])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny subset + tiny model, to prove the wiring")
    args = ap.parse_args()

    loader = SEVIRLoader(SEVIRConfig.from_store(
        Path(args.store),
        inputs=[DEFAULT_CHANNELS["ir107"], DEFAULT_CHANNELS["ir069"]],
        target=VIL_CHANNEL, context_frames=2, horizon_frames=6))

    report = split_sevir(loader)
    print(report.summary(), "\n")

    tr_ids = report["train"].event_ids
    va_ids = report["val"].event_ids
    va_days = report["val"].days
    if args.smoke:
        tr_ids, va_ids = tr_ids[:24], va_ids[:12]
        va_days = va_days[:12]
        args.epochs, args.dim, args.depth = 1, 16, 1

    print(f"computing channel stats on {min(len(tr_ids), 100)} TRAIN events "
          f"(train only -- stats over val/test would leak)")
    stats = compute_channel_stats(loader, tr_ids, n_sample=8 if args.smoke else 100)
    print(f"  mean {[round(m, 2) for m in stats.mean]}  "
          f"std {[round(s, 2) for s in stats.std]}\n")

    dcfg = DatasetConfig(nan_policy=NaNPolicy(args.nan_policy), stats=stats,
                         targets=TargetConfig())
    tr = SEVIRDataset(loader, tr_ids, dcfg)
    va = SEVIRDataset(loader, va_ids, dcfg)

    model = MultiTaskNowcaster(ModelConfig(
        in_channels=tr.in_channels, context_frames=2,
        grid_size=loader.cfg.grid_size, lead_steps=6,
        dim=args.dim, depth=args.depth))
    print(f"model: {model.n_parameters():,} params, "
          f"{tr.in_channels} input channels, heads {list(model.geometries)}")

    # Provenance stamped into every checkpoint. The cloudburst head is
    # trained on pseudo-stations sampled from the VIL grid: SEVIR has no gauge
    # network, so those weights train the sampling and sparse-scoring MECHANISM
    # and give a sensible initialisation. They carry NO real station
    # relationship -- that can only come from IMD AWS/ARG data at fine-tune
    # time. Anyone loading this checkpoint needs to know that.
    provenance = {
        "rain_rate": "SEVIR VIL > 1.0 kg/m^2, coarsened to 12 km",
        "extreme_rain": "SEVIR VIL > 25.6 kg/m^2 (~99.9th pct at 12 km)",
        "cloudburst": "MECHANISM ONLY -- pseudo-stations sampled from the VIL "
                      "grid. No real station relationship. Re-supervise on IMD "
                      "AWS/ARG data before any operational use.",
    }
    notes = ("SEVIR (CONUS) pretraining. Synoptically forced convection over "
             "flat terrain, no orographic analogue and no DEM. Provides a "
             "debugged pipeline and a warm start, NOT monsoon physics. No "
             "scientific claim rests on SEVIR numbers.")

    from nowcast_train import train
    train(model, tr, va,
          TrainConfig(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                      run_dir=Path(args.run_dir), device=args.device,
                      num_workers=args.workers,
                      notes=notes, head_provenance=provenance,
                      n_boot=100 if args.smoke else 500),
          lead_minutes=loader.cfg.lead_minutes,
          val_groups=episode_ids(va_days),
          resume=not args.no_resume)


if __name__ == "__main__":
    main()
