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

from .calibration import operating_points
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
    amp: bool = True            # mixed precision; ignored off CUDA
    wandb_project: str = ""     # empty disables W&B entirely
    wandb_run: str = ""
    # Validation cost is dominated by the FSS sweep: 10 thresholds x 4
    # neighbourhoods x 7 blocks = 280 filter passes, ~87 s per head on the
    # val set. During training only the headline threshold is needed, which
    # is ~10x cheaper; the full sweep belongs in the final report.
    lean_validation: bool = True
    # False-alarm-ratio ceiling for the deployable operating point. Selecting
    # on SEDI alone gave FAR 0.997 on the rare heads -- 997 false alarms per
    # 1000 warnings. SEDI's false-alarm term is the RATE b/(b+d), which stays
    # tiny when negatives dominate, so an unconstrained optimum buys recall at
    # almost any price. Both points are reported every epoch; neither is
    # quotable without the other.
    # PLACEHOLDER, not an analysis -- see docs/LIMITATIONS.md #6. A justified
    # ceiling follows from the cost ratio between a missed event and a false
    # evacuation, and is per-head because it depends on the base rate. 0.80 is
    # merely much better than the 0.997 an unconstrained optimum gives.
    far_ceiling: float = 0.80

    def __post_init__(self):
        self.run_dir = Path(self.run_dir)
        if self.device == "auto":
            self.device = ("cuda" if torch.cuda.is_available()
                           else "mps" if torch.backends.mps.is_available()
                           else "cpu")


def format_operating_points(ops: dict, metric: str = "sedi") -> str:
    """Both operating points side by side. Neither is quotable alone.

    The unconstrained point is the model's discrimination ceiling. The
    FAR-constrained point is what an operator could actually act on. Showing
    only the first is how a warning system with 997 false alarms per 1000
    ends up on a slide looking excellent.
    """
    lines = ["", "operating points (selected on validation)",
             f"{'head':>14} {'  UNCONSTRAINED':>30} {'  FAR-CONSTRAINED':>32}"]
    lines.append(f"{'':>14} {'thr':>8} {metric:>7} {'POD':>6} {'FAR':>6}"
                 f"{'thr':>10} {metric:>7} {'POD':>6} {'FAR':>6} {'ok?':>5}")
    lines.append("-" * 78)
    for head, d in ops.items():
        u, c = d["unconstrained"], d["far_constrained"]
        lines.append(
            f"{head:>14} {u['threshold']:>8.4f} {u[metric]:>7.3f} "
            f"{u['pod']:>6.3f} {u['far']:>6.3f}"
            f"{c['threshold']:>10.4f} {c[metric]:>7.3f} {c['pod']:>6.3f} "
            f"{c['far']:>6.3f} {('yes' if c['feasible'] else 'NO'):>5}")
    infeasible = [h for h, d in ops.items() if not d["far_constrained"]["feasible"]]
    if infeasible:
        lines.append(f"!!  no operating point meets FAR <= "
                     f"{list(ops.values())[0]['max_far']:.2f} for: "
                     f"{', '.join(infeasible)}. The least-bad point is shown; "
                     f"this head is not deployable at that ceiling yet.")
    return "\n".join(lines)


def head_eval_configs(model: MultiTaskNowcaster, grid_km: float = 4.0,
                      lean: bool = False, thresholds: dict | None = None) -> dict:
    """One EvalConfig per head, with the geometry the head actually emits.

    The cloudburst head is point geometry, so its scorecard omits FSS --
    scoring station output on a grid config would silently report a
    neighbourhood metric over station index.
    """
    out = {}
    for name, geom in model.geometries.items():
        thr = (thresholds or {}).get(name, 0.5)
        kw = dict(name=name, grid_km=grid_km, geometry=geom,
                  headline_threshold=float(thr))
        if lean:
            # one threshold, two neighbourhoods -- ~10x cheaper per epoch
            kw |= dict(thresholds=(float(thr),), neighborhood_km=(0.0, 50.0))
        out[name] = EvalConfig(**kw)
    return out


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
    # Raw arrays come back too, so the operating points can be computed
    # without a second inference pass.
    return result, order, preds, targs


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
    cfgs = head_eval_configs(model, lean=cfg.lean_validation)

    wb = None
    if cfg.wandb_project:
        try:
            import wandb
            wb = wandb.init(project=cfg.wandb_project,
                            name=cfg.wandb_run or cfg.run_dir.name,
                            config={k: str(v) for k, v in vars(cfg).items()},
                            resume="allow", id=cfg.wandb_run or None)
        except Exception as e:      # logging must never take down a run
            print(f"  W&B disabled ({type(e).__name__}: {e})")

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
        train_s = time.time() - t0        # training only, before validation
        train_loss = running / max(len(train_dl), 1)

        result, _, vprobs, vtargs = run_validation(
            model, val_dl, cfgs, lead_minutes, groups=val_groups,
            n_boot=cfg.n_boot, device=dev)
        name, score = result.worst_head_lower_bound(cfg.checkpoint_metric)
        ops = operating_points(vprobs, vtargs, metric=cfg.checkpoint_metric,
                               max_far=cfg.far_ceiling)

        epoch_s = time.time() - t0
        print(f"\nepoch {epoch}  train_loss {train_loss:.4f}  "
              f"train {train_s:.0f}s  val {epoch_s - train_s:.0f}s  "
              f"total {epoch_s:.0f}s")
        if wb is not None:
            wb.log({"epoch": epoch, "train_loss": train_loss,
                    "train_seconds": train_s, "val_seconds": epoch_s - train_s,
                    "checkpoint_metric": score,
                    **{f"{h}/sedi": result[h].pooled["headline"].get("sedi", float("nan"))
                       for h in result.names},
                    **{f"{h}/csi": result[h].pooled["headline"]["csi"]
                       for h in result.names}})
        print(result.summary_table().split("--- ")[0])
        print(format_operating_points(ops, cfg.checkpoint_metric))
        if previous is not None:
            print(result.compare(previous, metric=cfg.checkpoint_metric))

        # Always save `last` so a reclaim loses at most one epoch.
        stamp = {"train_loss": train_loss, "notes": cfg.notes,
                 "head_provenance": cfg.head_provenance,
                 "operating_points": ops}
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
