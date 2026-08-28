"""The training loop.

Gradients come from the LOSS. Decisions come from the HARNESS. Those are
different objects and this file keeps them separate:

    loss     differentiable, per batch, focal + masked
    harness  non-differentiable, per epoch, drives checkpointing

Checkpoint selection is `worst_head_lower_bound("sedi")` -- the worst head,
on a base-rate-independent metric, at its lower confidence bound, with the
bootstrap blocked on storm EPISODE. Each of those four words was chosen to
close a specific way of fooling yourself; see nowcast_eval/multihazard.py.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from nowcast_eval import EvalConfig, attach_confidence_intervals, evaluate_multi
from nowcast_model import MultiTaskLoss, MultiTaskNowcaster

from .checkpoint import find_latest, load_checkpoint, save_checkpoint
from .dataset import collate


@dataclass
class TrainConfig:
    epochs: int = 20
    batch_size: int = 8
    lr: float = 3e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    num_workers: int = 0
    device: str = "auto"
    run_dir: Path = Path("runs/default")
    n_boot: int = 500
    loss_weights: dict = field(default_factory=dict)
    checkpoint_metric: str = "sedi"
    log_every: int = 20
    # Free-text provenance stamped into every checkpoint. Used to record what
    # the weights do NOT mean -- e.g. that a head was trained on pseudo-labels.
    notes: str = ""
    head_provenance: dict = field(default_factory=dict)

    def __post_init__(self):
        self.run_dir = Path(self.run_dir)
        if self.device == "auto":
            self.device = ("cuda" if torch.cuda.is_available()
                           else "mps" if torch.backends.mps.is_available()
                           else "cpu")


def head_eval_configs(model: MultiTaskNowcaster, grid_km: float = 4.0) -> dict:
    """One EvalConfig per head, with the geometry the head actually emits.

    The cloudburst head is point geometry, so its scorecard omits FSS --
    scoring station output on a grid config would silently report a
    neighbourhood metric over station index.
    """
    return {name: EvalConfig(name=name, grid_km=grid_km, geometry=geom)
            for name, geom in model.geometries.items()}


@torch.no_grad()
def run_validation(model, loader, cfgs, lead_minutes, groups=None, n_boot=500,
                   device="cpu"):
    """Score the val set through the same harness code path as the final report."""
    model.eval()
    preds, targs = {k: [] for k in cfgs}, {k: [] for k in cfgs}
    order = []

    for x, y, m, coords, ids in loader:
        out = model(x.to(device), coords.to(device))
        for k, v in out.items():
            preds[k].append(torch.sigmoid(v).float().cpu().numpy())
            targs[k].append(y[k].numpy())
        order.extend(ids)

    preds = {k: np.concatenate(v) for k, v in preds.items()}
    targs = {k: np.concatenate(v) for k, v in targs.items()}

    result = evaluate_multi(preds, targs, cfgs, lead_minutes=lead_minutes)
    for name in result.names:
        attach_confidence_intervals(result[name], preds[name], targs[name],
                                    cfgs[name], n_boot=n_boot,
                                    groups=groups if groups is not None else None)
    return result, order


def train(model: MultiTaskNowcaster, train_ds, val_ds, config: TrainConfig,
          lead_minutes, val_groups=None, resume: bool = True):
    """Train, validating and checkpointing every epoch.

    `val_groups` is one storm-episode label per val EVENT, in dataset order.
    Without it the validation intervals assume independence between windows
    that share a synoptic setup, which on real SEVIR overstates the effective
    sample size by roughly two orders of magnitude.
    """
    cfg = config
    dev = torch.device(cfg.device)
    model = model.to(dev)

    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                          num_workers=cfg.num_workers, collate_fn=collate,
                          drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                        num_workers=cfg.num_workers, collate_fn=collate)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                            weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    criterion = MultiTaskLoss(weights=cfg.loss_weights)
    cfgs = head_eval_configs(model)

    start_epoch, best = 0, -float("inf")
    if resume:
        latest = find_latest(cfg.run_dir)
        if latest is not None:
            meta = load_checkpoint(latest, model=model, optimizer=opt, scheduler=sched)
            start_epoch, best = meta["epoch"] + 1, meta["best_score"]
            print(f"resumed from {latest} at epoch {start_epoch} "
                  f"(best {best:.4f})")

    previous = None
    for epoch in range(start_epoch, cfg.epochs):
        model.train()
        t0, running = time.time(), 0.0

        for i, (x, y, m, coords, _) in enumerate(train_dl):
            x, coords = x.to(dev), coords.to(dev)
            y = {k: v.to(dev) for k, v in y.items()}
            m = {k: v.to(dev) for k, v in m.items()}

            loss, per_head = criterion(model(x, coords), y, m)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()

            running += loss.item()
            if cfg.log_every and i % cfg.log_every == 0:
                print(f"  epoch {epoch} step {i}/{len(train_dl)} "
                      f"loss {loss.item():.4f} " +
                      " ".join(f"{k}={v.item():.4f}" for k, v in per_head.items()))

        sched.step()
        train_loss = running / max(len(train_dl), 1)

        result, _ = run_validation(model, val_dl, cfgs, lead_minutes,
                                   groups=val_groups, n_boot=cfg.n_boot,
                                   device=dev)
        name, score = result.worst_head_lower_bound(cfg.checkpoint_metric)

        print(f"\nepoch {epoch}  train_loss {train_loss:.4f}  "
              f"{time.time() - t0:.0f}s")
        print(result.summary_table().split("--- ")[0])
        if previous is not None:
            print(result.compare(previous, metric=cfg.checkpoint_metric))

        # Always save `last` so a reclaim loses at most one epoch.
        stamp = {"train_loss": train_loss, "notes": cfg.notes,
                 "head_provenance": cfg.head_provenance}
        save_checkpoint(cfg.run_dir / "last.pt", model=model, optimizer=opt,
                        scheduler=sched, epoch=epoch, best_score=best,
                        config=cfg, extra=stamp)

        if score == -float("inf"):
            print(f"!!  checkpoint BLOCKED: head {name!r} is unmeasurable "
                  f"({cfg.checkpoint_metric} undefined). Not saving 'best' -- "
                  f"a score computed while blind to a head is not a score.")
        elif score > best:
            best = score
            save_checkpoint(cfg.run_dir / "best.pt", model=model, optimizer=opt,
                            scheduler=sched, epoch=epoch, best_score=best,
                            config=cfg, extra=stamp | {"worst_head": name})
            print(f"  new best: {name} {cfg.checkpoint_metric} lower bound "
                  f"{best:.4f} -> saved best.pt")
        else:
            print(f"  no improvement ({name} {score:.4f} <= best {best:.4f})")

        # ALL heads, every epoch. Saving only the first head made the
        # threshold sweep look rain_rate-only and hid that the rare heads had
        # real skill below the 0.5 operating point.
        for hname in result.names:
            result[hname].to_json(
                str(cfg.run_dir / f"epoch{epoch:03d}_{hname}.json"))
        previous = result

    return {"best_score": best, "run_dir": str(cfg.run_dir)}
