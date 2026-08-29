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

# Enable Tensor Cores for RTX 50-series / 40-series
if torch.cuda.is_available():
    torch.set_float32_matmul_precision('high')

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
# These are approximate statistics derived from the IMDAA reanalysis
# and INSAT satellite data over the Aug 15-25, 2019 window.
# Variables are sorted alphabetically (matching data_loader ordering):
# HGT(6 levels), RH(6 levels), TMP(6 levels), UGRD(6 levels), VGRD(6 levels)

IMDAA_MEAN = torch.tensor([
    # HGT: 1000, 300, 500, 700, 850, 925 hPa (geopotential height, meters)
    100.0, 9300.0, 5500.0, 3000.0, 1450.0, 750.0,
    # RH: 1000, 300, 500, 700, 850, 925 hPa (relative humidity, %)
    70.0, 30.0, 45.0, 55.0, 65.0, 72.0,
    # TMP: 1000, 300, 500, 700, 850, 925 hPa (temperature, Kelvin)
    300.0, 228.0, 256.0, 275.0, 288.0, 295.0,
    # UGRD: 1000, 300, 500, 700, 850, 925 hPa (u-wind, m/s)
    0.0, 15.0, 5.0, 2.0, 1.0, 0.5,
    # VGRD: 1000, 300, 500, 700, 850, 925 hPa (v-wind, m/s)
    0.0, 5.0, 2.0, 1.0, 0.5, 0.2,
]).view(30, 1, 1, 1)

IMDAA_STD = torch.tensor([
    # HGT
    50.0, 200.0, 150.0, 100.0, 60.0, 40.0,
    # RH
    20.0, 25.0, 25.0, 22.0, 20.0, 18.0,
    # TMP
    8.0, 5.0, 6.0, 7.0, 7.0, 7.0,
    # UGRD
    5.0, 15.0, 10.0, 7.0, 5.0, 4.0,
    # VGRD
    4.0, 10.0, 7.0, 5.0, 4.0, 3.0,
]).view(30, 1, 1, 1)

# INSAT channels: WV brightness temp (~200-280 K), CTT (~180-300 K), HEM (~0-100)
INSAT_MEAN = torch.tensor([240.0, 240.0, 50.0]).view(3, 1, 1, 1)
INSAT_STD = torch.tensor([25.0, 30.0, 25.0]).view(3, 1, 1, 1)

# Terrain: Elevation (~0-7000 m), Slope (~0-60 deg)
TERRAIN_MEAN = torch.tensor([2500.0, 10.0]).view(2, 1, 1)
TERRAIN_STD = torch.tensor([2000.0, 12.0]).view(2, 1, 1)

# ============================================================
# Training Hyperparameters
# ============================================================
BATCH_SIZE = 8                  # Increased — GPU has headroom (4.9/16 GB at BS=4)
EPOCHS = 30
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 4                 # Parallel data loading workers
TRAIN_SPLIT = 0.8               # 80/20 train/val split

# Class imbalance weights for BCEWithLogitsLoss
# Cloudbursts and Flash Floods are rare compared to thunderstorms
POS_WEIGHTS = torch.tensor([50.0, 1.0, 50.0])

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
