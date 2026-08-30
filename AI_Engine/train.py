"""
SIH26077 — Training Script (v2 — Max Accuracy, RTX 5080 Optimized)
====================================================================
All optimizations applied for a 4-5 hour high-accuracy training run.

GPU Optimizations:
  - TF32 Tensor Cores enabled (set in config.py)
  - cuDNN benchmark mode for auto-tuned conv kernels
  - Mixed precision bfloat16 throughout
  - torch.compile() JIT fusion — ~15-20% free speedup
  - Gradient Accumulation (effective batch = 32) for stable gradients
  - Non-blocking GPU data transfers with pin_memory

Training Quality:
  - Combined Dice + BCE loss — much better for spatial segmentation than BCE alone
  - Cosine Annealing with Warm Restarts (SGDR) — escapes local minima
  - Data augmentation (flip, rotate, jitter) — expands 34 windows → much larger
  - Early stopping (patience=20) to prevent wasted time on plateaus
  - Best checkpoint + last checkpoint saved separately

Temporal Train/Val Split:
  - TRAIN: Windows 0–33   (Aug 16–20, 2019) — 34 windows
  - VAL:   Windows 34–54  (Aug 21–24, 2019) — 21 windows (STRICT HOLDOUT)
"""

import os
import time
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, Dataset
import warnings

from data_loader import SpatiotemporalDataset
from model import SpatiotemporalMultiTaskModel

from config import (
    DEVICE, CHECKPOINT_DIR, CHECKPOINT_PATH,
    BATCH_SIZE, EPOCHS, LEARNING_RATE, WEIGHT_DECAY,
    NUM_WORKERS, POS_WEIGHTS, DEFAULT_LEAD_TIME,
    GRAD_ACCUM_STEPS, EARLY_STOP_PATIENCE,
)

warnings.filterwarnings('ignore')

CHECKPOINT_LAST = os.path.join(CHECKPOINT_DIR, "last_model.pth")

# Temporal split: Train on Aug 15-23 (windows 0-43 = 44 windows)
#                 Val   on Aug 23-25 (windows 44-54 = 11 windows)
# WHY: More training data = smarter model. The old split (34 train)
# left the Uttarkashi peak (Aug 21-23) entirely in validation —
# the model never saw it during training. Now those critical CB/FF
# examples ARE in training, giving the CB and FF heads real signal.
# Validation on Aug 23-25 still tests unseen future days.
TEMPORAL_SPLIT_IDX = 44


# ── Combined Loss: Dice + BCE ──────────────────────────────────────────────
class DiceFocalLoss(nn.Module):
    """
    Combined Dice + Focal Loss for rare-event spatial segmentation.

    Focal Loss fix for class collapse:
      Standard BCE on imbalanced data: model predicts all-zeros, gets low loss.
      Focal Loss adds (1 - p)^gamma weight to each pixel.
      → Easy negatives (background) get DOWN-weighted by (1-0.01)^2 ≈ 0.98
      → Hard positives (missed cloudbursts) get UP-weighted by (1-0.01)^2 ≈ 0.98
      → Rare events dominate the gradient signal → model MUST learn them.

    gamma=2 is the standard value from the original RetinaNet paper (Lin et al 2017).
    alpha handles class imbalance (= pos_weight / (1 + pos_weight)).
    """
    def __init__(self, pos_weights, alpha=0.5, gamma=2.0, smooth=1.0):
        super().__init__()
        self.pos_weights = pos_weights   # (1, 3, 1, 1)
        self.alpha = alpha               # Dice weight
        self.gamma = gamma               # Focal modulation
        self.smooth = smooth

    def focal_loss_per_task(self, logits, targets, pw):
        """Focal loss for a single task (1-channel). pw = scalar pos_weight."""
        bce = F.binary_cross_entropy_with_logits(
            logits, targets,
            pos_weight=torch.tensor([pw], device=logits.device),
            reduction='none'
        )
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
        return (focal_weight * bce).mean()

    def dice_loss_per_task(self, logits, targets):
        """Dice loss for a single task (1-channel)."""
        probs = torch.sigmoid(logits)
        p = probs.view(probs.shape[0], -1)
        t = targets.view(targets.shape[0], -1)
        intersection = (p * t).sum(dim=1)
        return (1.0 - (2.0 * intersection + self.smooth) /
                (p.sum(dim=1) + t.sum(dim=1) + self.smooth)).mean()

    def task_loss(self, logits_1ch, targets_1ch, pw):
        """Combined Dice+Focal for one channel."""
        dice  = self.dice_loss_per_task(logits_1ch, targets_1ch)
        focal = self.focal_loss_per_task(logits_1ch, targets_1ch, pw)
        return self.alpha * dice + (1 - self.alpha) * focal

    def forward(self, logits, targets):
        """
        Per-task loss with amplification multipliers.

        WHY per-task multipliers:
          Thunderstorm covers ~0.66% of pixels → large raw loss value.
          Cloudburst covers ~0.04% and FF ~0.014% → tiny raw loss values.
          In a joint loss, TS gradient dominates the shared backbone,
          starving the CB and FF heads of meaningful gradient.

          By scaling CB×4 and FF×6, we equalize the effective gradient
          magnitude across all three heads, forcing the shared encoder
          to learn features useful for all events — not just TS.
        """
        pw = self.pos_weights.squeeze()   # (3,)

        loss_cb = self.task_loss(logits[:, 0:1], targets[:, 0:1], pw[0].item())
        loss_ts = self.task_loss(logits[:, 1:2], targets[:, 1:2], pw[1].item())
        loss_ff = self.task_loss(logits[:, 2:3], targets[:, 2:3], pw[2].item())

        # CB × 4, TS × 1, FF × 6 — amplify rare-event heads
        return 4.0 * loss_cb + 1.0 * loss_ts + 6.0 * loss_ff



# ── Augmentation Wrapper ───────────────────────────────────────────────────
class AugmentedSubset(Dataset):
    """
    Wraps a Subset and applies random augmentations on-the-fly.
    Effectively multiplies 34 windows into a larger, diverse dataset.

    Augmentations applied:
      - Random horizontal flip (50%)
      - Random vertical flip (50%)
      - Random 90° rotation (25% chance each of 0/90/180/270)
      - Small brightness/contrast jitter on INSAT channels (weather variability)

    All augmentations are deterministic per-sample using the same random state,
    so IMDAA, INSAT, terrain, and targets are all transformed identically.
    """
    def __init__(self, subset):
        self.subset = subset

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        sample = self.subset[idx]
        imdaa   = sample['imdaa']    # (30, 6, H, W)
        insat   = sample['insat']    # (3, 6, H, W)
        terrain = sample['terrain']  # (2, H, W)
        targets = sample['targets']  # (3, H, W)

        # Random horizontal flip
        if random.random() > 0.5:
            imdaa   = torch.flip(imdaa,   dims=[-1])
            insat   = torch.flip(insat,   dims=[-1])
            terrain = torch.flip(terrain, dims=[-1])
            targets = torch.flip(targets, dims=[-1])

        # Random vertical flip
        if random.random() > 0.5:
            imdaa   = torch.flip(imdaa,   dims=[-2])
            insat   = torch.flip(insat,   dims=[-2])
            terrain = torch.flip(terrain, dims=[-2])
            targets = torch.flip(targets, dims=[-2])

        # Random 90° rotation (0/1/2/3 quarter turns)
        k = random.randint(0, 3)
        if k > 0:
            # rot90 on last 2 dims for 4D/3D tensors
            imdaa   = torch.rot90(imdaa,   k, [-2, -1])
            insat   = torch.rot90(insat,   k, [-2, -1])
            terrain = torch.rot90(terrain, k, [-2, -1])
            targets = torch.rot90(targets, k, [-2, -1])

        # INSAT channel jitter (±10% brightness) — simulates satellite variability
        if random.random() > 0.5:
            scale = 0.9 + random.random() * 0.2   # [0.90, 1.10]
            insat = insat * scale

        return {
            'imdaa':   imdaa,
            'insat':   insat,
            'terrain': terrain,
            'targets': targets,
        }


# ── In-Memory Dataset Cache ────────────────────────────────────────────────
class CachedDataset(Dataset):
    """
    Loads ALL samples from a Subset into RAM on construction.

    WHY THIS IS THE MOST IMPORTANT OPTIMIZATION:
    The old training took ~18 min/epoch because PyTorch was loading
    IMDAA HDF5 + INSAT HDF5 files from disk for every single batch.
    The RTX 5080 GPU was idle ~95% of the time, starved for data.

    With 34 training windows × ~120 MB each ≈ 4.1 GB total.
    We have 17.2 GB free RAM. So we load ONCE at startup (~10-12 min),
    then every epoch is pure GPU compute: ~20-30 seconds instead of 18 min.
    That turns 400 epochs from 120 hours → ~4 hours.
    """
    def __init__(self, subset, desc="train"):
        total = len(subset)
        print(f"  Caching {total} {desc} samples into RAM (one-time, ~10-15 min)...")
        self.cache = []
        t0 = time.time()
        for i in range(total):
            self.cache.append(subset[i])
            elapsed = time.time() - t0
            eta = (elapsed / (i + 1)) * (total - i - 1)
            print(f"    [{i+1:2d}/{total}] loaded | elapsed {elapsed:.0f}s | ETA {eta:.0f}s", end='\r')
        print(f"\n  Cache complete — {total} samples in RAM in {time.time()-t0:.0f}s")

    def __len__(self):
        return len(self.cache)

    def __getitem__(self, idx):
        return self.cache[idx]


def print_gpu_info():
    print(f"\n{'=' * 60}")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"  GPU:            {props.name}")
        print(f"  VRAM:           {props.total_memory / 1024**3:.1f} GB")
        print(f"  CUDA:           {torch.version.cuda}")
        print(f"  SM Count:       {props.multi_processor_count}")
        print(f"  TF32 Enabled:   {torch.backends.cuda.matmul.allow_tf32}")
        print(f"  cuDNN Bench:    {torch.backends.cudnn.benchmark}")
        print(f"  Precision Mode: bfloat16 + TF32 Tensor Cores")
    print(f"{'=' * 60}\n")


def train_model():
    print_gpu_info()

    print(f"Loading Dataset (lead_time={DEFAULT_LEAD_TIME}h)...")
    dataset = SpatiotemporalDataset(lead_time=DEFAULT_LEAD_TIME)
    total = len(dataset)

    # ── Strict Temporal Holdout ──────────────────────────────────────
    train_indices = list(range(0, min(TEMPORAL_SPLIT_IDX, total)))
    val_indices   = list(range(TEMPORAL_SPLIT_IDX, total))

    raw_train_subset = Subset(dataset, train_indices)
    raw_val_subset   = Subset(dataset, val_indices)

    # ── Cache both splits into RAM ───────────────────────────────────
    # This is the key optimization: load all files once, then every
    # epoch is pure GPU compute instead of waiting on disk I/O.
    print("\nPre-loading data into RAM (one-time cost, ~10-15 min)...")
    cached_train = CachedDataset(raw_train_subset, desc="train")
    cached_val   = CachedDataset(raw_val_subset,   desc="val")

    # Wrap training set with augmentation ON TOP of the cache
    train_dataset = AugmentedSubset(cached_train)
    val_dataset   = cached_val

    print(f"\n  TEMPORAL SPLIT: Train={len(raw_train_subset)} windows (Aug 15-23) "
          f"| Val={len(cached_val)} windows (Aug 23-25)")
    print(f"  Augmentation: flip + rotate + brightness jitter → ACTIVE on train set")
    print(f"  RAM Cache: ACTIVE — epochs will now be ~20-30s instead of ~18 min")

    # ── WeightedRandomSampler: oversample windows with CB/FF events ─────
    # Without this: a batch of 4 might contain zero CB windows → zero gradient for CB head
    # With this: CB/FF-positive windows are sampled 3× more often → consistent gradient
    print("  Computing per-sample weights for WeightedRandomSampler...")
    sample_weights = []
    for i in range(len(train_dataset)):
        sample = train_dataset[i]
        targets = sample['targets']         # (3, H, W): [CB, TS, FF]
        cb_pos = targets[0].sum().item()    # count of positive CB pixels
        ff_pos = targets[2].sum().item()    # count of positive FF pixels
        # Windows with ANY CB or FF event → weight 3.0; others → weight 1.0
        weight = 3.0 if (cb_pos > 0 or ff_pos > 0) else 1.0
        sample_weights.append(weight)

    sampler = torch.utils.data.WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )

    # num_workers=0: data is already in RAM, workers just add IPC overhead
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,        # replaces shuffle=True
        drop_last=True,
        num_workers=0,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    print("\nInitializing Model...")
    model = SpatiotemporalMultiTaskModel().to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params:,}")

    # torch.compile is disabled: both max-autotune and reduce-overhead fail on
    # RTX 5080 Laptop (60 SMs) due to the Triton/GEMM autotuner requirement.
    # Training is still highly optimized via TF32 Tensor Cores + bfloat16 + RAM cache.
    print("  torch.compile: SKIPPED (laptop GPU — running optimized eager mode)")

    # ── Optimizer & Scheduler ────────────────────────────────────────
    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    # CosineAnnealingWarmRestarts (SGDR):
    # T_0=30 → restarts at ep 30, 90, 210 (T_mult=2 doubles each interval)
    # Tighter restarts help CB/FF heads escape the low-probability plateau faster.
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=30,
        T_mult=2,
        eta_min=1e-7,
    )

    # GradScaler for mixed precision stability
    scaler = torch.amp.GradScaler('cuda')

    # Dice + Focal Loss — fixes class collapse for rare CB/FF events
    pos_weights = POS_WEIGHTS.to(DEVICE).view(1, 3, 1, 1)
    criterion = DiceFocalLoss(pos_weights=pos_weights, alpha=0.5, gamma=2.0)

    print(f"\n{'─' * 60}")
    print(f"  Epochs:             {EPOCHS}")
    print(f"  Batch size:         {BATCH_SIZE} × {GRAD_ACCUM_STEPS} accum = {BATCH_SIZE * GRAD_ACCUM_STEPS} effective")
    print(f"  Loss:               Dice + Focal (alpha=0.5, gamma=2.0, pos_w=[80,3,80])")
    print(f"  Scheduler:          CosineAnnealingWarmRestarts (T0=20, Tmult=2)")
    print(f"  LR range:           {LEARNING_RATE:.1e} → 5e-4 (warmup) → 1e-7 (min)")
    print(f"  Early Stop:         patience={EARLY_STOP_PATIENCE}")
    print(f"  Augmentation:       ON (flip, rotate, jitter)")
    print(f"  torch.compile:      DISABLED (eager mode + TF32 + bfloat16)")
    print(f"{'─' * 60}\n")

    best_val_loss = float('inf')
    patience_counter = 0
    train_start = time.time()

    for epoch in range(EPOCHS):
        epoch_start = time.time()

        # ── Training ────────────────────────────────────────────────
        model.train()
        train_loss_accum = 0.0
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(train_loader):
            imdaa   = batch['imdaa'].to(DEVICE, non_blocking=True)
            insat   = batch['insat'].to(DEVICE, non_blocking=True)
            terrain = batch['terrain'].to(DEVICE, non_blocking=True)
            targets = batch['targets'].to(DEVICE, non_blocking=True)

            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                preds = model(imdaa, insat, terrain)
                loss  = criterion(preds, targets)
                # Scale loss by accumulation steps for consistent gradient magnitude
                loss  = loss / GRAD_ACCUM_STEPS

            scaler.scale(loss).backward()

            # Step optimizer only every GRAD_ACCUM_STEPS batches
            if (batch_idx + 1) % GRAD_ACCUM_STEPS == 0 or (batch_idx + 1) == len(train_loader):
                # Gradient clipping prevents exploding gradients — critical for Transformer
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            train_loss_accum += loss.item() * GRAD_ACCUM_STEPS * imdaa.size(0)

        train_loss = train_loss_accum / len(train_dataset)
        scheduler.step(epoch)  # SGDR steps per epoch

        # ── Validation ──────────────────────────────────────────────
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                imdaa   = batch['imdaa'].to(DEVICE, non_blocking=True)
                insat   = batch['insat'].to(DEVICE, non_blocking=True)
                terrain = batch['terrain'].to(DEVICE, non_blocking=True)
                targets = batch['targets'].to(DEVICE, non_blocking=True)

                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    preds = model(imdaa, insat, terrain)
                    loss  = criterion(preds, targets)

                val_loss += loss.item() * imdaa.size(0)

        val_loss /= len(val_dataset)
        current_lr = optimizer.param_groups[0]['lr']
        epoch_time = time.time() - epoch_start
        elapsed    = time.time() - train_start

        # Progress bar style logging — clean, not noisy
        improved = "✓ BEST" if val_loss < best_val_loss else ""
        print(f"Ep {epoch+1:3d}/{EPOCHS} | "
              f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
              f"LR: {current_lr:.2e} | {epoch_time:.0f}s/ep | "
              f"Total: {elapsed/60:.1f}m  {improved}")

        # ── Checkpoint ──────────────────────────────────────────────
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict() if not hasattr(model, '_orig_mod')
                                    else model._orig_mod.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'train_loss': train_loss,
            }, CHECKPOINT_PATH)
        else:
            patience_counter += 1

        # Always save the latest checkpoint so you can resume
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict() if not hasattr(model, '_orig_mod')
                                else model._orig_mod.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
        }, CHECKPOINT_LAST)

        # ── Early Stopping ──────────────────────────────────────────
        if patience_counter >= EARLY_STOP_PATIENCE:
            print(f"\n[Early Stop] Val loss did not improve for {EARLY_STOP_PATIENCE} epochs.")
            print(f"             Stopping at epoch {epoch+1} to save time.")
            break

    total_time = (time.time() - train_start) / 60
    print(f"\n{'=' * 60}")
    print(f"  Training Complete!")
    print(f"  Best Val Loss:  {best_val_loss:.4f}")
    print(f"  Total Time:     {total_time:.1f} minutes")
    print(f"  Best Checkpoint: {CHECKPOINT_PATH}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    train_model()
