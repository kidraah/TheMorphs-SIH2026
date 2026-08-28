"""Losses for extremely imbalanced targets.

The problem: at a base rate near 1e-4, a model that outputs "no" everywhere
achieves near-perfect accuracy and a very low plain BCE. Gradient descent
finds that solution quickly and sits there. The loss curve looks excellent
and the model is worthless -- which is why the harness scores SEDI and CSI
per epoch rather than trusting the loss.

Two standard remedies, both here:
  * pos_weight   -- scale up the loss on positive pixels
  * focal loss   -- down-weight examples already classified confidently, so
                    gradient budget goes to the hard, rare cases
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def focal_loss_with_logits(logits, targets, alpha: float = 0.75,
                           gamma: float = 2.0, reduction: str = "mean",
                           valid_mask=None):
    """Lin et al. 2017 focal loss.

    alpha weights the positive class (0.75 leans toward recall, which is
    usually right for a warning system: a missed cloudburst costs more than
    a false alarm). gamma controls how sharply easy examples are discounted.

    `valid_mask` excludes cells that are missing in the observations --
    satellite gaps decoded to NaN. Letting NaN into the loss produces NaN
    gradients and silently destroys the whole run.
    """
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce * ((1 - p_t) ** gamma)
    if alpha is not None:
        a_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = a_t * loss

    if valid_mask is not None:
        loss = loss * valid_mask
        if reduction == "mean":
            n = valid_mask.sum()
            return loss.sum() / n.clamp(min=1.0)
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss


class MultiTaskLoss(nn.Module):
    """Weighted sum over heads.

    On the weights: the three heads have base rates spanning orders of
    magnitude, so their raw losses are NOT comparable and an unweighted sum
    is dominated by whichever head has the most positives. That is the
    mechanism behind the MTL failure mode the scorecard watches for -- the
    shared backbone reallocates capacity toward the loudest gradient.

    Start at 1.0 each, then use `MultiHazardResult.compare()` between epochs
    to see which head is losing, and raise its weight. Tuning these blind is
    guesswork; tuning them against the per-head regression report is not.
    """

    def __init__(self, weights: dict[str, float] | None = None,
                 alpha: float = 0.75, gamma: float = 2.0):
        super().__init__()
        self.weights = weights or {}
        self.alpha, self.gamma = alpha, gamma

    def forward(self, logits: dict, targets: dict, masks: dict | None = None):
        if set(logits) != set(targets):
            raise ValueError(f"head mismatch: logits {sorted(logits)} vs "
                             f"targets {sorted(targets)}")
        masks = masks or {}
        per_head, total = {}, None
        for name, lg in logits.items():
            l = focal_loss_with_logits(lg, targets[name], self.alpha, self.gamma,
                                       valid_mask=masks.get(name))
            per_head[name] = l
            w = self.weights.get(name, 1.0)
            total = w * l if total is None else total + w * l
        return total, per_head
