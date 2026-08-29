# SIH26077 — AI-Driven Hyper-Local Early Warning System

> **Smart India Hackathon 2026** | Ministry of Earth Sciences (MoES)  
> Real-time severe weather nowcasting for cloudbursts, thunderstorms, and flash floods

## Overview

An advanced AI predictive engine that simultaneously predicts the onset of severe thunderstorms, cloudbursts, and flash floods with a **2–6 hour lead time**. The system fuses multi-modal atmospheric data (IMDAA reanalysis + INSAT-3DR satellite + CartoDEM terrain) through a **spatiotemporal deep learning architecture** with a multi-task learning approach.

## Architecture

```
IMDAA (30 channels × 6 timesteps)  ──┐
                                      ├─→ Temporal Compression (3D Conv)
INSAT (3 channels × 6 timesteps)   ──┘           │
                                                  ├─→ Early Fusion (112ch)
CartoDEM (elevation + slope)       ──────→ Embed ─┘          │
                                                       Shared U-Net Encoder
                                                     ┌───────┼───────┐
                                                     ▼       ▼       ▼
                                                  Cloudburst  TS   Flash Flood
                                                   Decoder  Decoder  Decoder
                                                     │       │       │
                                                     ▼       ▼       ▼
                                               3 × Probability Risk Maps (256×256)
```

## Project Structure

```
├── config.py              # Centralized paths, hyperparameters, thresholds
├── data_loader.py         # Multi-modal dataset with z-score normalization
├── model.py               # Spatiotemporal Multi-Task U-Net
├── train.py               # GPU-accelerated training (mixed precision)
├── predict.py             # Inference & risk map generation
├── xai.py                 # Explainable AI (GradCAM) module
├── alerts.py              # Automated threshold-based alerting
├── app.py                 # Streamlit interactive dashboard
├── run_pipeline.py        # End-to-end pipeline runner
├── analysis/              # EDA & precursor analysis scripts
│   ├── analyze_insat_signatures.py
│   ├── analyze_imdaa_thermodynamics.py
│   └── analyze_dem_topography.py
├── checkpoints/           # Saved model weights
├── dataset_root/          # Raw data (IMDAA, INSAT, DEM, targets)
├── output/                # Generated risk maps and plots
└── ps.md                  # Problem statement
```

## Quick Start

### Prerequisites
```
Python 3.11+
PyTorch 2.x with CUDA
```

### Install Dependencies
```bash
pip install torch torchvision xarray h5py rasterio numpy matplotlib streamlit folium streamlit-folium
```

### Run Full Pipeline
```bash
python run_pipeline.py
```

This will:
1. Verify data loader & spatial alignment
2. Test model architecture
3. Train the model (GPU accelerated, ~30 epochs)
4. Generate risk map predictions
5. Test XAI module
6. Test alerting engine
7. Launch the interactive web dashboard

### Run Individual Components
```bash
python data_loader.py   # Test data loading
python model.py         # Test model architecture
python train.py         # Train the model
python predict.py       # Generate risk maps
python xai.py           # Test GradCAM
python alerts.py        # Test alerts
streamlit run app.py    # Launch dashboard
```

## Data Sources

| Source | Format | Description |
|--------|--------|-------------|
| **IMDAA** | .nc (NetCDF) | Reanalysis: Temperature, Humidity, Wind at 6 pressure levels |
| **INSAT-3DR** | .h5 (HDF5) | Satellite: Water Vapor, Cloud Top Temperature, Humidity/Emissivity |
| **CartoDEM** | .tif (GeoTIFF) | 30m Digital Elevation Model: elevation + terrain slope |

## Key Technical Features

- **Multi-Task Learning**: Single forward pass → 3 simultaneous risk predictions
- **Mixed Precision**: bfloat16 training with GradScaler for RTX 50-series
- **Input Normalization**: Channel-wise z-score normalization for stable training
- **Parallel Data Loading**: Multi-worker DataLoader with pinned memory
- **Explainable AI**: GradCAM heatmaps showing model attention regions
- **Automated Alerts**: WATCH / WARNING / EMERGENCY severity classification
- **Interactive Dashboard**: Folium map overlays with real-time inference

## Team

**THE MORPHS** — Smart India Hackathon 2026
