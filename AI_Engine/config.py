"""
SIH26077 — Centralized Configuration
=====================================
Single source of truth for all paths, hyperparameters, thresholds,
and normalization statistics used across the entire pipeline.
"""

import os
import torch

# ============================================================
# Paths
# ============================================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Dataset
DATASET_ROOT = os.path.join(PROJECT_ROOT, "dataset_root")
INDEX_PATH = os.path.join(DATASET_ROOT, "window_index.json")
DEM_PATH = os.path.join(DATASET_ROOT, "dem", "cartodem.tif")

# Output
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pth")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# Device
# ============================================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Enable TF32 Tensor Cores — CRITICAL for RTX 50-series/40-series Blackwell/Ada
# Without this, matmul runs in full FP32, wasting ~3x compute on the Tensor Cores.
if torch.cuda.is_available():
    torch.set_float32_matmul_precision('high')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True   # cuDNN auto-tunes conv kernels per input size

# ============================================================
# Data
# ============================================================
GRID_SIZE = (256, 256)          # Unified spatial grid (H, W)
LEAD_TIMES = ['2', '3', '4', '5', '6']
DEFAULT_LEAD_TIME = '3'        # 3-hour prediction window

# IMDAA variables per timestep (5 variables × 6 pressure levels = 30 channels)
# Variables: HGT, RH, TMP, UGRD, VGRD
# Levels:    1000, 300, 500, 700, 850, 925 hPa
IMDAA_CHANNELS = 30
IMDAA_TIMESTEPS = 6

# INSAT channels: WV (L1B), CTT (L2B_CTP), HEM (L2B_HEM)
INSAT_CHANNELS = 3
INSAT_TIMESTEPS = 6

# Terrain: Elevation + Slope
TERRAIN_CHANNELS = 2

# Target: Cloudburst, Thunderstorm, Flash Flood
NUM_TARGETS = 3
TARGET_NAMES = ['Cloudburst', 'Thunderstorm', 'Flash Flood']

# ============================================================
# Normalization Statistics (channel-wise mean/std)
# ============================================================
# Channel ordering (after timestamp-based grouping in data_loader):
#   ch[00-05]  HGT  at 1000, 300, 500, 700, 850, 925 hPa  (geopotential height, m)
#   ch[06-11]  RH   at 1000, 300, 500, 700, 850, 925 hPa  (relative humidity, %)
#   ch[12-17]  TMP  at 1000, 300, 500, 700, 850, 925 hPa  (temperature, K)
#   ch[18-23]  UGRD at 1000, 300, 500, 700, 850, 925 hPa  (u-wind, m/s)
#   ch[24-29]  VGRD at 1000, 300, 500, 700, 850, 925 hPa  (v-wind, m/s)
#
# Values computed empirically from windows 0, 22, 44 using correct
# timestamp-grouped IMDAA files. Pressure levels sort alphabetically:
# 1000 < 300 < 500 < 700 < 850 < 925 mb (by leading digit).

IMDAA_MEAN = torch.tensor([
    # HGT (geopotential height, m): 1000, 300, 500, 700, 850, 925 hPa
      68.10, 9454.68, 5721.24, 3065.77, 1457.89,  739.89,
    # RH (relative humidity, %): 1000, 300, 500, 700, 850, 925 hPa
      64.14,   35.49,   44.89,   55.46,   60.53,   64.67,
    # TMP (temperature, K): 1000, 300, 500, 700, 850, 925 hPa
     294.63,  237.24,  261.38,  276.90,  286.31,  290.23,
    # UGRD (u-wind, m/s): 1000, 300, 500, 700, 850, 925 hPa
      -0.13,   -1.27,    0.28,    1.02,    0.68,    0.35,
    # VGRD (v-wind, m/s): 1000, 300, 500, 700, 850, 925 hPa
       1.87,   -0.49,   -0.70,   -0.16,    0.85,    1.86,
]).view(30, 1, 1, 1)

IMDAA_STD = torch.tensor([
    # HGT
      54.39, 1525.39,  922.82,  495.01,  238.00,  127.57,
    # RH
      25.87,   25.38,   29.89,   25.18,   26.01,   28.10,
    # TMP
      47.76,   38.35,   42.21,   44.76,   46.38,   47.07,
    # UGRD
       4.83,   11.51,    5.78,    4.97,    6.44,    6.32,
    # VGRD
       3.87,    7.33,    4.51,    3.99,    4.38,    4.73,
]).view(30, 1, 1, 1)


# INSAT channels: WV (raw counts ~700-1000), CTT (~180-320 K), HEM (0-257 mm/30min), CTT_RATE (K/step)
# CTT_RATE: frame-to-frame CTT difference. Mean≈0 (as many warming as cooling steps),
# Std≈5K/step during active monsoon convection.
INSAT_MEAN = torch.tensor([900.0, 265.0, 2.5, 0.0]).view(4, 1, 1, 1)
INSAT_STD  = torch.tensor([ 80.0,  30.0, 5.0, 5.0]).view(4, 1, 1, 1)

# Terrain: Elevation (~0-7000 m), Slope (~0-60 deg)
TERRAIN_MEAN = torch.tensor([2500.0, 10.0]).view(2, 1, 1)
TERRAIN_STD = torch.tensor([2000.0, 12.0]).view(2, 1, 1)

# ============================================================
# Training Hyperparameters  (Optimized for RTX 5080 Laptop, 4-5 hr run)
# ============================================================
BATCH_SIZE = 4
EPOCHS = 400
LEARNING_RATE = 1e-4            # Higher than before — needed to break CB/FF out of zero
WEIGHT_DECAY = 1e-3
NUM_WORKERS = 0
TRAIN_SPLIT = 0.8
GRAD_ACCUM_STEPS = 8
EARLY_STOP_PATIENCE = 50       # More patience — Focal Loss trains slower initially

# Class imbalance weights for BCE loss
# Updated after switching CB labels to HEM (2816x2805 resolution):
#   CB: HEM gives ~200 positive pixels/window (vs 25 from GPI) → imbalance ~300:1 → pos_weight=50
#   TS: ~430 positive pixels (~0.66%) → pos_weight=3
#   FF: regenerated slope-mask labels, ~47 positive pixels → pos_weight=80
POS_WEIGHTS = torch.tensor([50.0, 3.0, 80.0])

# ============================================================
# Alert Thresholds
# ============================================================
ALERT_THRESHOLDS = {
    'WATCH':     0.3,   # Probability > 30%
    'WARNING':   0.5,   # Probability > 50%
    'EMERGENCY': 0.7,   # Probability > 70%
}

# From label_config.json — thresholds used for generating weak labels
LABEL_THRESHOLDS = {
    'QPE_CLOUDBURST_THRESH_MM': 50.0,
    'CTT_THUNDERSTORM_THRESH_K': 208.15,
    'CAPE_THUNDERSTORM_THRESH': 1500.0,
}

# ============================================================
# Dashboard / Map
# ============================================================
MAP_BOUNDS = [[29.0, 76.0], [33.0, 81.0]]  # Himachal/Uttarakhand
MAP_CENTER = [31.0, 78.5]
