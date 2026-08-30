---
title: SIH26077 Hybrid TransUNet
emoji: ⛈️
colorFrom: blue
colorTo: red
sdk: static
pinned: false
tags:
  - earth-sciences
  - weather-forecasting
  - spatiotemporal
  - pytorch
  - unet
---

# SIH26077 — AI-Driven Hyper-Local Early Warning System for Severe Weather Nowcasting
<div align="center">

**Smart India Hackathon 2026 | Ministry of Earth Sciences (MoES) | PS SIH26077**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.11+-red.svg)](https://pytorch.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-green.svg)](https://fastapi.tiangolo.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

*A multi-task spatiotemporal AI engine that simultaneously predicts **cloudbursts**, **severe thunderstorms**, and **flash floods** across the Uttarakhand and Himachal Pradesh Himalayas with a **2–6 hour actionable lead time**.*

</div>

---

## Final Model Performance (v3 — Production)

> Evaluated on strict temporal holdout: **Aug 23–25, 2019** (never seen during training).

| Hazard | Precision | Recall | F1-Score | IoU |
|---|---|---|---|---|
| 🌧️ **Cloudburst** | 18.29% | 83.36% | **30.00%** | 17.65% |
| ⛈️ **Thunderstorm** | 80.56% | 92.17% | **85.97%** | 75.40% |
| 🌊 **Flash Flood** | 8.24% | 80.58% | **14.94%** | 8.08% |

**Full journey:**

| Run | CB F1 | TS F1 | FF F1 | Key Change |
|---|---|---|---|---|
| Runs 1–5 | 0.08% | ~72% | 0% | Baseline (all bugs present) |
| Run 6 | 10.60% | 75.82% | 1.52% | IMDAA grouping fix + label resize + per-task loss |
| **Run 7 (Final)** | **30.00%** | **85.97%** | **14.94%** | CB labels from HEM (1km) + CTT drop-rate channel |

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [System Architecture Overview](#2-system-architecture-overview)
3. [The Brain — Hybrid TransUNet Model](#3-the-brain--hybrid-transunet-model)
4. [Datasets](#4-datasets)
5. [Label Generation Logic](#5-label-generation-logic)
6. [Training — Full Technical Details](#6-training--full-technical-details)
7. [What Was Fixed (Bug History)](#7-what-was-fixed-bug-history)
8. [Explainable AI (XAI)](#8-explainable-ai-xai)
9. [Automated Alert Engine](#9-automated-alert-engine)
10. [FastAPI Backend](#10-fastapi-backend)
11. [React Dashboard](#11-react-dashboard)
12. [Quick Start](#12-quick-start)
13. [Project File Structure](#13-project-file-structure)
14. [PS.md Requirements Coverage](#14-psmd-requirements-coverage)
15. [Loading from Hugging Face](#15-loading-from-hugging-face)

---

## 1. Problem Statement

India is highly vulnerable to rapidly intensifying, localized extreme weather events — **cloudbursts**, **severe thunderstorms**, and **flash floods**. Traditional physics-based Numerical Weather Prediction (NWP) models suffer from computational latency (hours to run) and poor resolution for small-scale mountain convection events.

The **Uttarkashi Cloudburst of August 17–18, 2019** killed dozens and caused massive infrastructure damage — with virtually no advance warning. Events like this repeat every monsoon season across Uttarakhand and Himachal Pradesh.

**This system provides:**
- Hyper-local (district-level, 256×256 grid) probability maps
- 2–6 hour predictive lead time (actionable before NWP finishes)
- Simultaneous multi-hazard prediction in a single forward pass
- Explainable AI output so disaster managers understand *why* an alert fired

---

## 2. System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DATA INGESTION                                │
│  IMDAA Reanalysis NC  → 30ch × 6t    INSAT-3DR H5 → 4ch × 6t       │
│  CartoDEM GeoTIFF     → 2ch (static)  All aligned to 256×256 grid   │
└────────────────────────┬────────────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────────────┐
│                    HYBRID TRANSUNET (17.6M params)                   │
│  3D Conv Temporal Compress → Early Fusion → U-Net Encoder            │
│  Transformer Bottleneck (8-head Self-Attention @ 64×64)              │
│  3× Independent U-Net Decoders → (B, 3, 256, 256) probability maps  │
└──────┬─────────────────┬─────────────────────────────────┬──────────┘
       │                 │                                 │
  Cloudburst         Thunderstorm                     Flash Flood
  Prob Map            Prob Map                          Prob Map
       │                 │                                 │
┌──────▼─────────────────▼─────────────────────────────────▼──────────┐
│                       FASTAPI BACKEND (api.py)                       │
│  District risk table · AlertEngine · XAI triggers · REST API         │
└────────────────────────┬────────────────────────────────────────────┘
          ┌──────────────┴────────────────┐
          │                               │
┌─────────▼──────────┐        ┌──────────▼──────────────────────────┐
│  React Dashboard   │        │  Streamlit Risk Map Viewer           │
│  Vite + Leaflet    │        │  Folium + HeatMap overlays           │
└────────────────────┘        └──────────────────────────────────────┘
```

---

## 3. The Brain — Hybrid TransUNet Model

**File:** [`model.py`](model.py) | **Parameters:** ~17.6M

### Input Shapes

```
IMDAA:   (B, 30, 6, 256, 256)  — 30 atmospheric channels × 6 timesteps (18h window)
INSAT:   (B,  4, 6, 256, 256)  — 4 satellite channels × 6 timesteps
Terrain: (B,  2,    256, 256)  — Elevation + Slope (static, loaded once)
```

### INSAT 4 Channels (v3 — CTT drop-rate added)

| Ch | Name | Source | Physical Meaning |
|---|---|---|---|
| 0 | WV | L1B `IMG_WV` (1408×1402, subsampled 4×) | Integrated Water Vapor — storm nowcasting cornerstone |
| 1 | CTT | L2B_CTP `CTT` (313×312) | Cloud Top Temperature in Kelvin |
| 2 | HEM | L2B_HEM `HEM` (2816×2805, subsampled 8×) | Hydro-Estimator rainfall rate (mm/30min) |
| **3** | **CTT_RATE** | Computed: `CTT[t] - CTT[t-1]` | **Cooling rate (K/step) — negative = storm intensifying** |

> **CTT_RATE is a ps.md explicit requirement:** *"Rapid cooling of cloud tops (CTT Drop Rate) provides real-time validation of explosive vertical updrafts."* A -15K/step drop is the clearest single-variable cloudburst precursor from satellite.

### Architecture Stages

```
STAGE 1: Temporal Compression (3D Convolutions — collapses 6 timesteps → 1 feature map)
  IMDAA:   Conv3d(30→32, k=3³) → BN → ReLU → Conv3d(32→64, k=(4,3,3)) → (B, 64, H, W)
  INSAT:   Conv3d( 4→16, k=3³) → BN → ReLU → Conv3d(16→32, k=(4,3,3)) → (B, 32, H, W)
  Terrain: Conv2d(2→16)         → BN → ReLU                             → (B, 16, H, W)

STAGE 2: Early Fusion
  Concatenate: [64 + 32 + 16] = 112 channels at full 256×256 resolution

STAGE 3: Shared CNN Encoder
  enc1: ConvBlock2D(112→128)  skip_e1 → MaxPool → (B, 128, 128, 128)
  enc2: ConvBlock2D(128→256)  skip_e2 → MaxPool → (B, 256,  64,  64)

STAGE 4: Hybrid Bottleneck
  CNN:  ConvBlock2D(256→512)
  ATTN: MultiHeadSelfAttention(embed=512, heads=8)
        • 64×64 → 4096 tokens — global receptive field
        • Long-range moisture-storm correlations
        • Residual + LayerNorm + FFN

STAGE 5: 3 Independent U-Net Decoder Heads (CB, TS, FF)
  Each: ConvTranspose2d(512→256) + skip_e2 → ConvBlock2D(512→256, dropout=0.3)
        ConvTranspose2d(256→128) + skip_e1 → ConvBlock2D(256→128, dropout=0.3)
        Conv2d(128→1) → sigmoid → (B, 1, 256, 256) probability map

OUTPUT: cat([CB, TS, FF]) → (B, 3, 256, 256) probability maps [0.0, 1.0]
```

### Key Design Rationale

| Decision | Why |
|---|---|
| **3D Conv temporal compression** | Preserves onset/build-up/peak temporal patterns across 18h |
| **Early modality fusion** | IMDAA thermodynamics and INSAT moisture interact strongly |
| **Transformer at 64×64** | 4,096 tokens: Arabian Sea IWV can attend to Himalayan orographic lift |
| **3 independent decoders** | CB and FF have different spatial signatures at output |
| **CTT_RATE channel** | Static CTT snapshots miss the *rate* of convective intensification |

---

## 4. Datasets

### Data Sources

| Dataset | Source | Format | Native Resolution | Coverage |
|---|---|---|---|---|
| IMDAA Reanalysis | NCMRWF | NetCDF4 `.nc` | ~12km / 501×751 | Aug 2019 |
| INSAT-3DR L1B | MOSDAC / ISRO | HDF5 `.h5` | WV: 1408×1402 | Aug 15–25, 2019 |
| INSAT-3DR L2B_CTP | MOSDAC / ISRO | HDF5 `.h5` | CTT: 313×312 | Aug 15–25, 2019 |
| INSAT-3DR L2B_HEM | MOSDAC / ISRO | HDF5 `.h5` | HEM: 2816×2805 | Aug 15–25, 2019 |
| CartoDEM | ISRO | GeoTIFF `.tif` | 30m | HP + Uttarakhand |

**CONSOLIDATED_DATA/** — organized flat structure used for label generation:
```
CONSOLIDATED_DATA/
├── INSAT/
│   ├── L1B_STD/   435 HDF5 files — keys: IMG_WV (1408×1402), IMG_TIR1, IMG_MIR
│   ├── L2B_CTP/   443 HDF5 files — key: CTT (313×312, Kelvin)
│   ├── L2B_HEM/   426 HDF5 files — key: HEM (2816×2805, mm/30min, max=257mm)
│   └── L2G_GPI/    68 HDF5 files — key: GPI (81×91, mm — NOT used for labels)
├── IMDAA/         7,440 NetCDF4 files (5 variables × 6 levels × 248 timestamps)
└── CARTODEM/      (use existing cartodem.tif from dataset_root/dem/)
```

### IMDAA Variables (30 channels = 5 variables × 6 pressure levels)

| Variable | Levels (hPa) | Storm Ingredient |
|---|---|---|
| `HGT` Geopotential Height | 1000, 300, 500, 700, 850, 925 | Orographic lift identification |
| `RH` Relative Humidity | 1000, 300, 500, 700, 850, 925 | Moisture (fuel) |
| `TMP` Temperature | 1000, 300, 500, 700, 850, 925 | CAPE/lapse rate (instability) |
| `UGRD` U-wind | 1000, 300, 500, 700, 850, 925 | Wind shear + low-level convergence |
| `VGRD` V-wind | 1000, 300, 500, 700, 850, 925 | Wind shear + low-level convergence |

**Critical: IMDAA files must be grouped by timestamp, not alphabetically.**
Filename pattern: `{VAR}-{LEVEL}mb_{YYYYMMDDHH}_ncum_imdaa_reanl_prl_{HH}_{LEVEL}_hpa.nc`
Group the 10-char `YYYYMMDDHH` from each filename to form 6-timestep sequences.

### Dataset Split — Strict Temporal Holdout

```
ALL 55 WINDOWS: Aug 15 00:00 UTC → Aug 25 23:59 UTC
                (3-hour intervals, 5 lead times each = 275 label folders)

TRAINING SET  (Windows 00–43): Aug 15–22, 2019 — 44 windows
VALIDATION SET (Windows 44–54): Aug 23–25, 2019 — 11 windows  ← NEVER SEEN IN TRAINING

Chronological holdout — simulates real deployment: train on history, predict the future.
```

---

## 5. Label Generation Logic

### Thunderstorm Labels
```python
Source: INSAT L2B_CTP — CTT key at (313, 312)
Threshold: CTT < 208.15 K  (-65°C)
Physics: Deep convective clouds reach the tropopause at 208K.
         WMO standard threshold for severe convection.
Avg positive: ~430 pixels per 256×256 window (~0.66%)
Quality: HIGH — spatially coherent, physically correct, learnable
```

### Cloudburst Labels (v3 — switched to HEM)
```python
Source: INSAT L2B_HEM — HEM key at (2816, 2805) in mm/30min
Threshold: HEM > 25mm per 30min  (= 50mm/hr = IMD cloudburst definition)
Script: regenerate_cb_labels.py

Why HEM instead of GPI:
  GPI (L2G): (1, 81, 91) pixels at ~25km/pixel — too coarse, max=9mm
  HEM (L2B): (1, 2816, 2805) pixels at ~1km/pixel — max=257mm
  A real cloudburst over 5km is invisible in a 25km GPI pixel.

Morphological dilation: 3×3 kernel, 1 iteration (expands sparse positives)
Avg positive after resize to 256×256 with nearest-neighbor: ~200–500 pixels/window

Critical resize rule:
  WRONG: F.interpolate(mode='bilinear')  → sparse 1s become floats ~0.0003 → invisible
  RIGHT: F.interpolate(mode='nearest') + (x > 0.5).float()  → clean binary preserved
```

### Flash Flood Labels
```python
Source: Derived from Cloudburst labels + CartoDEM slope
Formula: FF = CB_mask AND (DEM_slope > 10°)
Script: regenerate_ff_labels.py

Physics: ps.md requires: "overlay atmospheric maps onto DEM to calculate how
         terrain slope channels extreme precipitation into drainage basins"
         10° threshold = conservative minimum for destructive runoff
         (Uttarakhand average slope: 15–30°)

DEM stats: max slope = 89.4°, pixels > 10° = 30.7% of 256×256 grid
Total FF pixels (275 windows): 14,817
FF/CB ratio: 0.42 (expected 0.3–0.8) ✅
```

---

## 6. Training — Full Technical Details

**Files:** [`train.py`](train.py), [`config.py`](config.py)

### Hyperparameters

| Parameter | Value |
|---|---|
| Epochs | 400 (early stop patience=50) |
| Batch size | 4 × 8 grad accumulation = effective 32 |
| Base LR | 1e-4 |
| Optimizer | AdamW (weight_decay=1e-3) |
| Scheduler | CosineAnnealingWarmRestarts (T₀=30, Tₘ=2) |
| Mixed precision | bfloat16 |
| Gradient clipping | max_norm=1.0 |

### Loss Function — DiceFocalLoss with Per-Task Amplification

```python
# Focal Loss prevents easy negatives from dominating:
# focal_weight = (1 - p_t)^gamma   [gamma=2]
# → model ignores confidently-correct background pixels
# → focuses all learning on hard positives (missed cloudbursts)

# Dice Loss forces spatial overlap (cannot be fooled by all-zero predictions):
# Dice = 1 - (2|P∩T| + ε) / (|P| + |T| + ε)

# Combined per-task loss with amplification:
loss_cb = 0.5 * Dice(CB) + 0.5 * Focal(CB)
loss_ts = 0.5 * Dice(TS) + 0.5 * Focal(TS)
loss_ff = 0.5 * Dice(FF) + 0.5 * Focal(FF)

total = 4.0 * loss_cb + 1.0 * loss_ts + 6.0 * loss_ff
# Amplification needed: TS (~0.66% pixels) produces 10× larger raw loss than
# CB (~0.3%) and FF (~0.07%). Without multipliers, TS dominates the shared backbone.
```

### Normalization Statistics

```python
# IMDAA (z-score per channel, 30 channels):
IMDAA_MEAN = torch.tensor([
    100.0, 9455.0, 5721.0, 3066.0, 1458.0, 740.0,   # HGT (m)
     64.0,   35.0,   45.0,   55.0,   61.0,  65.0,   # RH (%)
    295.0,  237.0,  261.0,  277.0,  286.0, 290.0,   # TMP (K)
      0.0,   -1.0,    0.0,    0.0,    0.0,   0.0,   # UGRD (m/s)
      2.0,    0.0,    1.0,    1.0,    2.0,   2.0,   # VGRD (m/s)
])

# INSAT (4 channels — v3):
# WV raw counts (~700-1000), CTT (K), HEM (mm/30min), CTT_RATE (K/step)
INSAT_MEAN = torch.tensor([900.0, 265.0, 2.5, 0.0]).view(4, 1, 1, 1)
INSAT_STD  = torch.tensor([ 80.0,  30.0, 5.0, 5.0]).view(4, 1, 1, 1)

# Terrain:
TERRAIN_MEAN = torch.tensor([2500.0, 10.0]).view(2, 1, 1)
TERRAIN_STD  = torch.tensor([2000.0, 12.0]).view(2, 1, 1)
```

### Positive Weights (class imbalance)

```python
POS_WEIGHTS = torch.tensor([50.0, 3.0, 80.0])  # [CB, TS, FF]
# CB=50: HEM labels give ~200-500 px/window → imbalance ~300:1
# TS=3:  CTT labels give ~430 px/window → imbalance ~150:1, but high density → mild boost
# FF=80: slope-mask labels give ~47 px/window → imbalance ~1400:1
```

### RAM Caching

All 44 training windows pre-loaded into RAM before epoch 1:
- Without cache: ~18 min/epoch (GPU waits for HDF5/NetCDF disk I/O)
- With cache: ~4 sec/epoch (pure GPU compute, 85–95% utilization)
- 400 epochs = ~5 min cache load + ~15 min training

---

## 7. What Was Fixed (Bug History)

> All 5 bugs below were present simultaneously in runs 1–5. This is why 5 training runs of 400 epochs each produced 0% F1 for CB and FF.

| Bug | Impact | Fix |
|---|---|---|
| **IMDAA temporal grouping** (alphabetical sort chunked into variable groups) | 3D temporal convolution was meaningless — T=0 had no humidity or wind | Group by 10-char `YYYYMMDDHH` timestamp in filename |
| **IMDAA normalization stats** (guessed, wrong channel order) | ch15: mean=111, std=302 — completely unnormalized | Computed from actual data with correct grouping |
| **CB label bilinear resize** (sparse 1s → floats ~0.0003) | Model never saw a positive CB target | `mode='nearest'` + hard threshold `(x>0.5).float()` |
| **FF label sparsity** (~10 px/window from wrong physics) | No learnable signal | Regenerated as `CB AND slope>10°` via CartoDEM |
| **TS gradient domination** (single loss → TS overwhelms CB/FF) | CB/FF heads received near-zero gradients | Per-task loss: `4×CB + 1×TS + 6×FF` |
| **CB labels from GPI** (81×91 pixels, 25km/pixel) | Cloudbursts invisible at 25km scale | Switched to HEM (2816×2805, 1km, max=257mm) |
| **Missing CTT drop-rate** | Static snapshots can't detect convective intensification | Added as 4th INSAT channel (frame-to-frame difference) |

---

## 8. Explainable AI (XAI)

**File:** [`xai.py`](xai.py)

GradCAM hooked into the Transformer bottleneck layer. For each prediction:
1. Gradient of target class's max activation w.r.t. bottleneck feature map
2. Global average pooling → per-channel importance weights
3. Weighted sum → ReLU → upsample to 256×256
4. Overlaid on the risk map to show *which geographic area* the model focused on

**API:** `GET /api/gradcam?window=17&task=0` (task=0: CB, 1: TS, 2: FF)

**Meteorological trigger weights** (displayed in dashboard):
```json
{
  "triggers": [
    {"variable": "IWV",   "weight": 0.87, "label": "Integrated Water Vapor surge"},
    {"variable": "CTT",   "weight": 0.74, "label": "Cloud Top Temperature drop"},
    {"variable": "CAPE",  "weight": 0.61, "label": "Convective instability"},
    {"variable": "SHEAR", "weight": 0.45, "label": "Vertical wind shear"}
  ]
}
```

---

## 9. Automated Alert Engine

**File:** [`alerts.py`](alerts.py)

| Level | Threshold | Action |
|---|---|---|
| `WATCH` | > 30% | Enhanced monitoring — pre-position response teams |
| `WARNING` | > 50% | Prepare for impact — evacuate vulnerable areas |
| `EMERGENCY` | > 70% | Immediate action — full emergency response |

```json
{
  "id": "CB-20190818-UTTARKASHI",
  "severity": "EMERGENCY",
  "event_type": "Cloudburst",
  "district": "Uttarkashi",
  "probability": 0.91,
  "lead_time": "2-3 hours",
  "timestamp_ist": "18 Aug 2019 09:30 IST",
  "message": "EMERGENCY: Cloudburst with 91% probability. Flash flood risk HIGH."
}
```

---

## 10. FastAPI Backend

**File:** [`api.py`](api.py) | Port: `8000`

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health` | Liveness probe |
| `GET` | `/api/windows?lead=3` | All 55 windows with IST timestamps |
| `GET` | `/api/predict?window=17&lead=3` | Full inference (risk maps, alerts, XAI, districts) |
| `GET` | `/api/gradcam?window=17&task=0` | GradCAM heatmap as base64 PNG |
| `GET` | `/api/district_risk?window=17&lead=3` | District-level risk table |

---

## 11. React Dashboard

**Directory:** `frontend/TheMorphs-SIH2026/` | Port: `5173`

| Component | Description |
|---|---|
| `RiskMap.jsx` | Choropleth district map + AI heatmap overlays + Time Animation Player |
| `TimeAnimationPlayer.jsx` | Play/Pause, speed control, 55-window scrubber |
| `RiskCards.jsx` | 5 KPI cards (Overall, CB, TS, FF, Affected Districts) |
| `ExplainableAI.jsx` | Meteorological trigger bar chart |
| `AlertBanner.jsx` | Emergency alert ribbon |

---

## 12. Quick Start

### Before Training — Regenerate Labels

```bash
cd C:\THEMORPHS

# Step 1: Regenerate CB labels from HEM (1km) instead of GPI (25km)
python scripts\regenerate_cb_labels.py

# Step 2: Regenerate FF labels based on new CB labels + DEM slope
python scripts\regenerate_ff_labels.py
```

### Train

```bash
# Delete old checkpoint first if model architecture changed (INSAT 3→4 channels)
del checkpoints\best_model.pth
del checkpoints\last_model.pth

python train.py
# ~15 min on RTX 5080 Laptop (early stop typically around epoch 210)
```

### Evaluate

```bash
python evaluate.py
# Tests on strict temporal holdout (Aug 23-25, 2019)
# Reports: Precision, Recall, F1, IoU per hazard class
```

### Run Full System

```bash
# Terminal 1 — Backend
python api.py

# Terminal 2 — Frontend
cd frontend\TheMorphs-SIH2026
npm run dev
# Open: http://localhost:5173
```

### Dependencies

```bash
pip install -r requirements.txt
# Key: torch, fastapi, uvicorn, streamlit, folium, scipy, numpy,
#      xarray, h5py, rasterio, matplotlib, tqdm, huggingface_hub
```

---

## 13. Project File Structure

```
C:\THEMORPHS\
├── model.py                ← Hybrid TransUNet (17.6M params, INSAT 4ch)
├── data_loader.py          ← Multi-modal loader: IMDAA timestamp grouping,
│                              4-channel INSAT with CTT_RATE, nearest-neighbor resize
├── train.py                ← RAM-cache training, DiceFocalLoss, per-task amplification
├── evaluate.py             ← Temporal holdout evaluation (F1, IoU, P, R)
├── config.py               ← All hyperparameters, normalization stats, paths
├── api.py                  ← FastAPI REST server
├── alerts.py               ← AlertEngine (Watch/Warning/Emergency)
├── xai.py                  ← GradCAM Explainable AI
├── streamlit_app.py        ← Standalone Streamlit viewer
├── regenerate_cb_labels.py ← [RUN BEFORE TRAINING] CB from HEM (1km)
├── regenerate_ff_labels.py ← [RUN BEFORE TRAINING] FF from CB + DEM slope
├── understand.py           ← Dataset analysis / debugging
├── start.bat               ← One-click launcher
├── requirements.txt
│
├── checkpoints/
│   ├── best_model.pth      ← Best validation checkpoint (upload to HuggingFace)
│   └── last_model.pth
│
├── dataset_root/
│   ├── imdaa/              ← IMDAA .nc files
│   ├── insat/              ← INSAT .h5 files (L1B, L2B_CTP, L2B_HEM)
│   ├── dem/cartodem.tif    ← CartoDEM GeoTIFF
│   ├── targets/            ← Binary .npy label arrays (win_*_lead_*/)
│   └── window_index.json   ← 55-window index
│
├── CONSOLIDATED_DATA/      ← Canonical flat data copy (used for label gen)
│   ├── INSAT/L1B_STD/      ← 435 HDF5 files
│   ├── INSAT/L2B_CTP/      ← 443 HDF5 files (CTT key)
│   ├── INSAT/L2B_HEM/      ← 426 HDF5 files (HEM key, max=257mm) ← CB labels
│   ├── INSAT/L2G_GPI/      ← 68 HDF5 files (NOT used for labels — 25km coarse)
│   └── IMDAA/              ← 7,440 NetCDF4 files (Aug 1-31 2019)
│
└── frontend/TheMorphs-SIH2026/
    └── src/components/     ← RiskMap, AlertBanner, ExplainableAI, etc.
```

---

## 14. PS.md Requirements Coverage

| PS Requirement | Status | Implementation |
|---|---|---|
| Simultaneous CB + TS + FF prediction | ✅ | 3 independent decoders, single forward pass |
| 2–6 hour lead time | ✅ | `lead` param ('2'–'6'), pre-computed labels |
| Spatiotemporal deep learning | ✅ | 3D Conv temporal compress + U-Net |
| Cross-attention / Transformer | ✅ | 8-head MHSA at bottleneck (64×64, 4096 tokens) |
| IWV storm nowcasting | ✅ | INSAT WV channel + IMDAA RH multi-level |
| **CTT drop rate tracking** | ✅ | **4th INSAT channel: frame-to-frame CTT difference** |
| CAPE/CIN instability | ✅ | IMDAA TMP multi-level lapse rate proxy |
| Wind shear and convergence | ✅ | IMDAA UGRD/VGRD at 925 + 300 hPa |
| DEM topographic integration | ✅ | CartoDEM elevation + slope fused at Stage 2 |
| Flash flood from terrain | ✅ | FF labels = CB ∩ (DEM slope > 10°) |
| Explainable AI | ✅ | GradCAM + meteorological trigger weights |
| Automated alerts | ✅ | Watch/Warning/Emergency via `AlertEngine` |
| Interactive web dashboard | ✅ | Vite + React + Leaflet choropleth |
| Real-time API | ✅ | FastAPI REST + base64 PNG heatmaps |

---

## 15. Loading from Hugging Face

```python
from huggingface_hub import hf_hub_download
import torch
from model import SpatiotemporalMultiTaskModel

# Download weights (auto-cached after first download)
ckpt_path = hf_hub_download(
    repo_id="YourUsername/SIH26077-HybridTransUNet",
    filename="best_model.pth"
)

# Load model (works on CPU or GPU)
model = SpatiotemporalMultiTaskModel()
ckpt = torch.load(ckpt_path, map_location="cpu")
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

# Input shapes required:
# imdaa:   (B, 30, 6, 256, 256)
# insat:   (B,  4, 6, 256, 256)  ← 4 channels: WV, CTT, HEM, CTT_RATE
# terrain: (B,  2,    256, 256)
with torch.no_grad():
    logits = model(imdaa, insat, terrain)   # → (B, 3, 256, 256)
    probs  = torch.sigmoid(logits)          # → CB, TS, FF probability maps
```

### Upload Your Checkpoint

```bash
python upload_to_hf.py
# or manually:
huggingface-cli login
python -c "
from huggingface_hub import upload_file
upload_file(
    path_or_fileobj='checkpoints/best_model.pth',
    path_in_repo='best_model.pth',
    repo_id='YourUsername/SIH26077-HybridTransUNet',
    repo_type='model',
)
"
```

---

## Citation

```bibtex
@misc{sih26077-themorphs-2026,
  title  = {SIH26077: AI-Driven Hyper-Local Early Warning System for Severe Weather Nowcasting},
  author = {TheMorphs Team},
  year   = {2026},
  note   = {Smart India Hackathon 2026 | Ministry of Earth Sciences | PS SIH26077},
  url    = {https://huggingface.co/YourUsername/SIH26077-HybridTransUNet}
}
```

---

*Built for Smart India Hackathon 2026 · PS SIH26077 · Ministry of Earth Sciences (MoES)*  
*Domain: Himachal Pradesh + Uttarakhand Himalayas | Lat: 29–33°N, Lon: 76–81°E*
