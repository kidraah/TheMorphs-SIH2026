"""
SIH26077 — Training Script
============================
Trains the Spatiotemporal Multi-Task U-Net on IMDAA + INSAT + DEM data.

Key features:
  - Mixed precision (bfloat16) for RTX 5080
  - Multi-worker DataLoader with pinned memory
  - CosineAnnealing LR scheduler
  - Best-model checkpointing on validation loss
  - Per-epoch wall-clock timing
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from data_loader import SpatiotemporalDataset
from model import SpatiotemporalMultiTaskModel
import warnings

from config import (
    DEVICE, CHECKPOINT_DIR, CHECKPOINT_PATH,
    BATCH_SIZE, EPOCHS, LEARNING_RATE, WEIGHT_DECAY,
    NUM_WORKERS, TRAIN_SPLIT, POS_WEIGHTS, DEFAULT_LEAD_TIME,
)

warnings.filterwarnings('ignore')


def print_gpu_info():
    """Print GPU diagnostics at startup."""
    print(f"\n{'─' * 50}")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"  GPU Device:    {props.name}")
        print(f"  VRAM:          {props.total_memory / 1024**3:.1f} GB")
        print(f"  CUDA Version:  {torch.version.cuda}")
        print(f"  PyTorch:       {torch.__version__}")
        print(f"  Compute Mode:  Mixed Precision (bfloat16)")
    else:
        print("  [!] CUDA NOT AVAILABLE — training on CPU (will be very slow)")
    print(f"  Device:        {DEVICE}")
    print(f"{'─' * 50}\n")


def train_model():
    print_gpu_info()
    
    print(f"Loading Dataset (lead_time={DEFAULT_LEAD_TIME}h)...")
    dataset = SpatiotemporalDataset(lead_time=DEFAULT_LEAD_TIME)
    
    # 80/20 train/val split
    indices = torch.randperm(len(dataset)).tolist()
    train_size = int(TRAIN_SPLIT * len(dataset))
    train_dataset = Subset(dataset, indices[:train_size])
    val_dataset = Subset(dataset, indices[train_size:])
    
    print(f"  Train: {len(train_dataset)} windows | Val: {len(val_dataset)} windows")
    
    # DataLoader with parallel workers + pinned memory for GPU transfer
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
    )
    
    print("Initializing Model...")
    model = SpatiotemporalMultiTaskModel().to(DEVICE)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params:,}")
    
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler = torch.amp.GradScaler('cuda')
    
    # Class imbalance: cloudbursts & flash floods are rare events
    pos_weights = POS_WEIGHTS.to(DEVICE).view(1, 3, 1, 1)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
    
    print(f"\nStarting Training for {EPOCHS} epochs...")
    print(f"  Batch size: {BATCH_SIZE} | LR: {LEARNING_RATE} | Scheduler: CosineAnnealing")
    print(f"  Workers: {NUM_WORKERS} | Pin Memory: True\n")
    
    best_val_loss = float('inf')
    
    for epoch in range(EPOCHS):
        epoch_start = time.time()
        
        # ── Training ────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        
        for batch_idx, batch in enumerate(train_loader):
            imdaa = batch['imdaa'].to(DEVICE, non_blocking=True)
            insat = batch['insat'].to(DEVICE, non_blocking=True)
            terrain = batch['terrain'].to(DEVICE, non_blocking=True)
            targets = batch['targets'].to(DEVICE, non_blocking=True)
            
            optimizer.zero_grad()
            
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                preds = model(imdaa, insat, terrain)
                loss = criterion(preds, targets)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item() * imdaa.size(0)
            
            if batch_idx % 2 == 0:
                print(f"  Epoch {epoch+1}/{EPOCHS} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")
                
        train_loss /= len(train_dataset)
        
        # ── Validation ──────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                imdaa = batch['imdaa'].to(DEVICE, non_blocking=True)
                insat = batch['insat'].to(DEVICE, non_blocking=True)
                terrain = batch['terrain'].to(DEVICE, non_blocking=True)
                targets = batch['targets'].to(DEVICE, non_blocking=True)
                
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    preds = model(imdaa, insat, terrain)
                    loss = criterion(preds, targets)
                    
                val_loss += loss.item() * imdaa.size(0)
                
        val_loss /= len(val_dataset)
        
        # Step scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        epoch_time = time.time() - epoch_start
        print(f"═══ Epoch {epoch+1} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
              f"LR: {current_lr:.2e} | Time: {epoch_time:.1f}s ═══")
        
        # Checkpoint best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            print(f"  [OK] Saved new best model (val_loss={val_loss:.4f})")

    print(f"\n{'═' * 50}")
    print(f"  Training Complete. Best val_loss: {best_val_loss:.4f}")
    print(f"  Checkpoint: {CHECKPOINT_PATH}")
    print(f"{'═' * 50}")


if __name__ == "__main__":
    train_model()
