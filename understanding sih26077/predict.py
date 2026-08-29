"""
SIH26077 — Inference & Risk Map Generation
=============================================
Loads the trained model, runs inference on a selected time window,
and generates probability risk maps for all three hazard types.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from data_loader import SpatiotemporalDataset
from model import SpatiotemporalMultiTaskModel
import warnings

from config import (
    DEVICE, CHECKPOINT_PATH, OUTPUT_DIR, TARGET_NAMES, DEFAULT_LEAD_TIME,
)

warnings.filterwarnings('ignore')


def plot_risk_maps(cb_pred, ts_pred, ff_pred, save_path=None):
    """Generate and save predicted probability risk maps.
    
    Args:
        cb_pred: (H, W) cloudburst probability map [0, 1]
        ts_pred: (H, W) thunderstorm probability map [0, 1]
        ff_pred: (H, W) flash flood probability map [0, 1]
        save_path: Where to save the output PNG
    """
    if save_path is None:
        save_path = os.path.join(OUTPUT_DIR, "risk_maps.png")
    
    # Custom colormap: transparent green → yellow → opaque red
    colors = [(0, 1, 0, 0), (1, 1, 0, 0.5), (1, 0, 0, 1)]
    cmap_risk = LinearSegmentedColormap.from_list("risk", colors)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    maps = [cb_pred, ts_pred, ff_pred]
    titles = [
        f"Cloudburst Probability (Lead: {DEFAULT_LEAD_TIME}h)",
        f"Severe Thunderstorm Probability (Lead: {DEFAULT_LEAD_TIME}h)",
        f"Flash Flood Probability (Lead: {DEFAULT_LEAD_TIME}h)",
    ]
    
    for ax, risk_map, title in zip(axes, maps, titles):
        im = ax.imshow(risk_map, cmap=cmap_risk, vmin=0, vmax=1)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.axis('off')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.suptitle("AI-Generated Hyper-Local Risk Maps — SIH26077", 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Saved risk maps to {save_path}")


def run_inference(window_idx=17):
    """Run inference on a single time window.
    
    Args:
        window_idx: Index of the window to predict (default: 17, Aug 18 disaster)
    """
    print(f"Loading Model on device: {DEVICE}")
    model = SpatiotemporalMultiTaskModel().to(DEVICE)
    
    if os.path.exists(CHECKPOINT_PATH):
        try:
            model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=True))
            print("  [OK] Loaded trained checkpoint.")
        except Exception as e:
            print(f"  [!] Could not load checkpoint: {e}. Using untrained weights.")
    else:
        print("  [!] No checkpoint found. Using untrained weights for demonstration.")
        
    model.eval()
    
    print(f"Loading test window (index={window_idx})...")
    dataset = SpatiotemporalDataset(lead_time=DEFAULT_LEAD_TIME)
    sample = dataset[window_idx]
    
    # Add batch dimension and move to GPU
    imdaa = sample['imdaa'].unsqueeze(0).to(DEVICE)
    insat = sample['insat'].unsqueeze(0).to(DEVICE)
    terrain = sample['terrain'].unsqueeze(0).to(DEVICE)
    
    print("Running Inference...")
    with torch.no_grad():
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            preds = model(imdaa, insat, terrain)
        probs = torch.sigmoid(preds)  # Convert logits → probabilities
        
    # Extract the 3 maps: [CB, TS, FF]
    cb_map = probs[0, 0].float().cpu().numpy()
    ts_map = probs[0, 1].float().cpu().numpy()
    ff_map = probs[0, 2].float().cpu().numpy()
    
    print(f"  CB max prob: {cb_map.max():.4f}")
    print(f"  TS max prob: {ts_map.max():.4f}")
    print(f"  FF max prob: {ff_map.max():.4f}")
    
    print("Plotting Risk Maps...")
    plot_risk_maps(cb_map, ts_map, ff_map)
    print("Inference Pipeline Complete.")


if __name__ == "__main__":
    run_inference()
