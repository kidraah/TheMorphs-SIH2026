"""How the harness attaches to training. Read this before writing the model.

The point: you do NOT train on the harness. You train on a LOSS -- a
differentiable, per-batch number that produces gradients. The harness is
non-differentiable and runs once per epoch on data the model has not seen.
It never touches a gradient. What it drives is *decisions*:

    - which checkpoint you keep
    - when you stop
    - which hyperparameter run won
    - what goes on the slide

This distinction matters practically. Loss going down does not mean the
model caught more storms; with a base rate near 1e-2, loss can improve
steadily while the model quietly learns to predict "no" everywhere. The
harness is what catches that, which is why it exists before the model does.

Fixing the contract now -- the model emits (L, H, W) probabilities -- means
the validation path and the final report run the same code, so they cannot
drift apart.

Torch is imported lazily so this file stays readable without a GPU box.
"""
from __future__ import annotations

import numpy as np

from nowcast_eval import EvalConfig, evaluate


def validate_epoch(model, val_loader, config: EvalConfig,
                   lead_minutes=None, device="cuda"):
    """Run the model over the validation set and score it. Returns an
    EvaluationResult -- print `.summary_table()` after every epoch."""
    import torch

    model.eval()
    preds, obss = [], []
    with torch.no_grad():
        for x, y in val_loader:
            p = model(x.to(device))          # (B, L, H, W) logits
            p = torch.sigmoid(p)             # -> probabilities
            preds.append(p.float().cpu().numpy())
            obss.append(y.numpy())

    return evaluate(np.concatenate(preds), np.concatenate(obss),
                    config, lead_minutes=lead_minutes)


def checkpoint_metric(result) -> float:
    """The single number to select checkpoints on.

    Deliberately NOT validation loss, and deliberately NOT pixel CSI.

    FSS at 50 km, averaged over the 2-6 h leads, because:
      - FSS gives partial credit for a near-miss, which is what an operator
        actually cares about; pixel CSI would reject a forecast that is
        right about the storm and 20 km off about the pixel.
      - restricting to 2-6 h stops a model that is merely excellent at
        30 min persistence from winning -- that window is the project's
        entire claim.

    Change this if you like, but change it *deliberately* and write down
    why. Whatever sits here is what your model will be optimised toward.

    For the three-head MTL model use `multi_head_checkpoint_metric` below
    instead -- this function scores one head and would happily let the
    cloudburst head rot while the thunderstorm head carries the number.
    """
    target_km = 50.0
    t = result.config["headline_threshold"]
    vals = []
    for l in result.per_lead:
        if l.lead_minutes is None or not (120 <= l.lead_minutes <= 360):
            continue
        row = next((f for f in l.fss
                    if f["neighborhood_km"] == target_km and f["threshold"] == t), None)
        if row and np.isfinite(row["fss"]):
            vals.append(row["fss"])
    return float(np.mean(vals)) if vals else float("nan")


def multi_head_checkpoint_metric(multi_result) -> float:
    """Checkpoint metric for the three-head model.

    Three deliberate choices, each of which changes which epoch you keep:

    1. The WORST head, not the mean. An average is gameable -- a run can
       improve it by getting better at thunderstorms while losing
       cloudbursts entirely, the one outcome the project cannot accept.

    2. SEDI, not CSI. CSI falls with the base rate whatever the forecast
       quality: at identical quality across heads it spreads more than 8x
       and ranks the rarest head worst every time. A CSI minimum is a
       base-rate detector. SEDI is base-rate independent, so the minimum
       means something.

    3. The LOWER CONFIDENCE BOUND, not the point estimate. At a 2e-4 base
       rate the point estimate is noise; selecting on it selects the epoch
       whose validation draw was luckiest, which does not generalise. The
       lower bound asks what you can defend.

    Requires attach_confidence_intervals() on each head first. Returns
    -inf if any head is unmeasurable -- that blocks checkpointing rather
    than quietly selecting on the heads that happen to have data.
    """
    _, value = multi_result.worst_head_lower_bound("sedi")
    return value


def training_loop_sketch():
    """The shape of it. Not runnable -- the model does not exist yet."""
    return """
    cfg  = EvalConfig(grid_km=4.0, headline_threshold=0.5, name="run-017")
    best = -inf

    for epoch in range(n_epochs):
        for x, y in train_loader:
            loss = focal_loss(model(x), y)   # <- gradients come from HERE
            loss.backward(); opt.step(); opt.zero_grad()

        result = evaluate_multi(preds, obss, cfgs, LEAD_MINUTES)
        for h in result.names:               # groups = storm day per case
            attach_confidence_intervals(result[h], preds[h], obss[h],
                                        cfgs[h], groups=val_days)
        print(result.summary_table())        # <- decisions come from HERE
        print(result.compare(last_epoch))    # <- did any head regress?
        score = multi_head_checkpoint_metric(result)

        if score > best:
            best = score
            torch.save(model.state_dict(), "best.pt")
            result.to_json(f"artifacts/epoch{epoch:03d}.json")
    """


if __name__ == "__main__":
    print(__doc__)
    print(training_loop_sketch())
