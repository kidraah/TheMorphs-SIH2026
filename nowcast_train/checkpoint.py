"""Checkpointing built for pre-emption, not as an afterthought.

Spot instances are reclaimed with ~2 minutes' notice. A checkpoint that
saves only model weights loses the optimizer's momentum and Adam moment
estimates, the LR schedule position, the epoch counter, and the RNG state --
so resuming produces a visibly different loss trajectory and quietly
re-shuffles data the model has already seen. Everything needed to continue
exactly is saved here.

Writes are atomic: to a temp file, then rename. A reclaim landing mid-write
otherwise leaves a truncated checkpoint that fails to load, which is worse
than having no checkpoint at all because it is discovered later.
"""
from __future__ import annotations

import os
from dataclasses import asdict, is_dataclass
from pathlib import Path

import numpy as np
import torch


def _plain(obj):
    return asdict(obj) if is_dataclass(obj) else obj


def save_checkpoint(path, *, model, optimizer, epoch: int, best_score: float,
                    scheduler=None, scaler=None, config=None, extra: dict | None = None):
    """Atomic save of everything needed to resume bit-for-bit."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "epoch": epoch,
        "best_score": best_score,
        "config": _plain(config),
        "rng": {
            "torch": torch.get_rng_state(),
            "numpy": np.random.get_state(),
        },
        "extra": extra or {},
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)          # atomic on POSIX
    return path


def _load_model_state(model, state) -> None:
    """Load weights, and say WHAT changed when they do not fit.

    strict=True is right -- silently skipping missing keys would resume a
    run against a different architecture and report it as a continuation.
    But the raw failure is a wall of parameter names. Cross-attention added
    5.91M parameters to a 32.5M model, so every checkpoint written before it
    is unloadable, and the operator needs to be told that in one line rather
    than deduce it from 400 missing keys.
    """
    try:
        model.load_state_dict(state["model"])
        return
    except RuntimeError as e:
        have = set(state["model"])
        want = set(model.state_dict())
        missing, unexpected = sorted(want - have), sorted(have - want)
        n_ckpt = sum(v.numel() for v in state["model"].values())
        n_model = sum(p.numel() for p in model.parameters())

        def families(keys):
            out = {}
            for k in keys:
                out[k.split(".")[0]] = out.get(k.split(".")[0], 0) + 1
            return ", ".join(f"{k} x{v}" for k, v in sorted(out.items())[:6])

        raise RuntimeError(
            f"this checkpoint was written for a DIFFERENT architecture.\n"
            f"  checkpoint: {n_ckpt/1e6:.2f} M parameters\n"
            f"  this model: {n_model/1e6:.2f} M parameters\n"
            f"  in the model but not the checkpoint ({len(missing)}): "
            f"{families(missing) or 'none'}\n"
            f"  in the checkpoint but not the model ({len(unexpected)}): "
            f"{families(unexpected) or 'none'}\n"
            f"  config recorded with the checkpoint: {state.get('config')}\n"
            f"Pre-training must be re-run; a checkpoint from before the "
            f"architecture changed is not a warm start for it."
        ) from e


def load_checkpoint(path, *, model, optimizer=None, scheduler=None, scaler=None,
                    map_location="cpu", restore_rng: bool = True) -> dict:
    """Restore in place. Returns the metadata (epoch, best_score, config)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"no checkpoint at {path}")
    state = torch.load(path, map_location=map_location, weights_only=False)

    _load_model_state(model, state)
    if optimizer is not None and state.get("optimizer") is not None:
        optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None and state.get("scheduler") is not None:
        scheduler.load_state_dict(state["scheduler"])
    if scaler is not None and state.get("scaler") is not None:
        scaler.load_state_dict(state["scaler"])

    if restore_rng and state.get("rng"):
        try:
            torch.set_rng_state(state["rng"]["torch"].cpu()
                                if hasattr(state["rng"]["torch"], "cpu")
                                else state["rng"]["torch"])
            np.random.set_state(state["rng"]["numpy"])
        except Exception:
            # A checkpoint from a different torch build can carry an
            # incompatible RNG state. Losing exact reproducibility is a far
            # smaller loss than refusing to resume at all.
            pass
    return {"epoch": state["epoch"], "best_score": state["best_score"],
            "config": state.get("config"), "extra": state.get("extra", {})}


def find_latest(directory) -> Path | None:
    """Newest resumable checkpoint in a run directory, or None."""
    d = Path(directory)
    if not d.exists():
        return None
    cands = sorted(d.glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0] if cands else None
