"""
SIH26077 — Model Evaluation Script
====================================
Calculates detailed metrics (IoU, Precision, Recall, F1-Score)
to verify the trained Hybrid TransUNet is actually learning.

Evaluation uses the STRICT TEMPORAL HOLDOUT set:
  - Val Set: Windows 34–54 = Aug 21–24, 2019 (21 windows)
  - These dates were NEVER seen by the model during training.
  - This simulates a real early-warning deployment scenario.
"""

import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from data_loader import SpatiotemporalDataset
from model import SpatiotemporalMultiTaskModel
from config import DEVICE, CHECKPOINT_PATH, BATCH_SIZE, NUM_WORKERS, TRAIN_SPLIT

# Per-class thresholds calibrated from diagnostic:
#   Cloudburst:   max=0.6694 → threshold=0.50
#   Thunderstorm: max=0.9841 → threshold=0.50
#   Flash Flood:  max=0.6824 → threshold=0.50
CLASS_THRESHOLDS = [0.50, 0.50, 0.50]   # [Cloudburst, Thunderstorm, Flash Flood]

def calculate_metrics(preds, targets, thresholds=CLASS_THRESHOLDS):
    """Calculate TP, FP, FN for a batch using per-class thresholds."""
    thresh = torch.tensor(thresholds, device=preds.device).view(1, 3, 1, 1)
    preds_binary  = (preds > thresh).float()
    targets_binary = targets.float()

    tp = (preds_binary * targets_binary).sum(dim=(0, 2, 3))
    fp = (preds_binary * (1 - targets_binary)).sum(dim=(0, 2, 3))
    fn = ((1 - preds_binary) * targets_binary).sum(dim=(0, 2, 3))

    return tp, fp, fn

def evaluate_model():
    print(f"Loading Model from {CHECKPOINT_PATH}...")
    model = SpatiotemporalMultiTaskModel().to(DEVICE)
    try:
        ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        state = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state)
    except FileNotFoundError:
        print("[!] No trained model found. Run train.py first.")
        return
        
    model.eval()
    
    print("Loading Validation Dataset (Aug 21-24, 2019 — temporal holdout)...")
    dataset = SpatiotemporalDataset(lead_time='3')
    
    # STRICT TEMPORAL HOLDOUT: same split as train.py
    # Windows 44-54 = Aug 23-25 2019 — never seen in training.
    TEMPORAL_SPLIT_IDX = 44
    val_indices = list(range(TEMPORAL_SPLIT_IDX, len(dataset)))
    val_subset  = Subset(dataset, val_indices)
    loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    print(f"  Evaluating on {len(val_subset)} windows (Aug 23-25, 2019 peak days)")
    
    total_tp = torch.zeros(3).to(DEVICE)
    total_fp = torch.zeros(3).to(DEVICE)
    total_fn = torch.zeros(3).to(DEVICE)
    
    print("Running Evaluation...")
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            imdaa = batch['imdaa'].to(DEVICE)
            insat = batch['insat'].to(DEVICE)
            terrain = batch['terrain'].to(DEVICE)
            targets = batch['targets'].to(DEVICE)
            
            # Forward pass
            logits = model(imdaa, insat, terrain)
            probs = torch.sigmoid(logits)
            
            tp, fp, fn = calculate_metrics(probs, targets)
            total_tp += tp
            total_fp += fp
            total_fn += fn

    # Calculate final metrics per class
    classes = ["Cloudburst", "Thunderstorm", "Flash Flood"]
    
    print("\n" + "="*50)
    print(f" FINAL EVALUATION METRICS")
    print(f" Thresholds — CB: {CLASS_THRESHOLDS[0]} | TS: {CLASS_THRESHOLDS[1]} | FF: {CLASS_THRESHOLDS[2]}")
    print("="*50)
    
    for i, class_name in enumerate(classes):
        tp = total_tp[i].item()
        fp = total_fp[i].item()
        fn = total_fn[i].item()
        
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
        iou = tp / (tp + fp + fn + 1e-8)
        
        print(f" {class_name.upper()}")
        print(f"   Precision: {precision*100:.2f}% (When it predicts {class_name}, how often is it right?)")
        print(f"   Recall:    {recall*100:.2f}% (Out of all real {class_name}s, how many did it catch?)")
        print(f"   F1-Score:  {f1*100:.2f}% (Overall balance)")
        print(f"   IoU:       {iou*100:.2f}% (Spatial overlap accuracy)")
        print("-" * 50)

if __name__ == "__main__":
    evaluate_model()
