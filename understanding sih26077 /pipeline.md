# SIH26077 — Working Prototype Pipeline

## 1. Registration checklist (do these today, in parallel with prototyping)

| Portal | For | Link | Notes |
|---|---|---|---|
| NCMRWF RDS | IMDAA reanalysis | https://rds.ncmrwf.gov.in | Sign Up -> email approval (can take days) |
| MOSDAC | INSAT-3D/3DR WV+TIR, GSMaP rain | https://mosdac.gov.in | Sign Up -> email approval. **Note:** WV/TIR channel data is *not* in the no-login "Open Data" section — only GSMaP-type rain products are; WV/TIR needs the full "Order Data" approval flow |
| Bhuvan/NOEDA | CartoDEM | https://bhuvan-app3.nrsc.gov.in/data/download | Free registration |
| **Copernicus CDS** | ERA5 (fallback for WV *and* cloud-top-temp) | https://cds.climate.copernicus.eu | Free, **instant** — no approval wait. Do this one first, it unblocks two fallbacks at once |
| **Kaggle** | BharatBench (fallback for IMDAA) | https://kaggle.com | Free account; click Download on the dataset page, or use `kagglehub` |
| **NASA Earthdata** | GPM IMERG (fallback for precip) | https://urs.earthdata.nasa.gov | Free, instant |

While MOSDAC/NCMRWF/Bhuvan are pending, `data/fetch_data.py` defaults to the CDS/Kaggle/Earthdata/SRTM fallbacks above — none of which require an approval wait — so you can build and test the full pipeline today.

## 2. Dataset map: what the PS asks for vs. what we substitute meanwhile

| Need (from PS) | Official India source | Access hurdle | Fallback used now | Access hurdle | Swap back via |
|---|---|---|---|---|---|
| Thermodynamics: CAPE/CIN, humidity, geopotential, wind | IMDAA reanalysis (NCMRWF) | Approval wait (days) | **BharatBench** — IMDAA pre-regridded, hosted on Kaggle | Free account, instant | `fetch_thermodynamics(..., use_official=True)` |
| Moisture / IWV | INSAT-3D/3DR Water Vapor channel (MOSDAC) | Approval wait — WV/TIR is behind "Order Data", *not* in Open Data | **ERA5 total column water vapour (TCWV)** via Copernicus CDS | Free API key, instant | `fetch_satellite_ir_wv(..., use_official=True)` |
| Cloud-top temperature | INSAT-3D/3DR TIR channel (MOSDAC) | Same as above — also behind approval, not open | **ERA5 top net thermal radiation (OLR)** via the *same* CDS key — colder cloud tops ↔ lower outgoing longwave radiation, so it tracks the same updraft signal | Free API key, instant (same one as the row above) | `fetch_satellite_ir_wv(..., use_official=True)` |
| Precipitation (QPE) | GSMaP ISRO Rain (MOSDAC Open Data) | Basic MOSDAC login only, no long approval | **NASA GPM IMERG** half-hourly | Free Earthdata account, instant | `fetch_precip(..., use_official=True)` |
| DEM / terrain | CartoDEM v3, 30m (Bhuvan/NOEDA) | Free registration | **SRTM 30m** via the `elevation` package | No login at all | `fetch_dem(use_official=True)` |
| Labeled severe-weather events | *(none published)* | — | Weak labels derived from physical thresholds (CAPE percentile, precip rate) — see `train.py` docstring | — | Replace with IMD bulletin-derived event dates when compiled |

**Key correction from earlier:** moisture (IWV) and cloud-top-temperature both now come from the *same* ERA5/CDS pull — register once at `cds.climate.copernicus.eu`, and both fallbacks are unblocked together. There is no genuinely open substitute for INSAT's actual TIR channel over India; OLR is the closest physically-motivated proxy, not a like-for-like replacement — worth stating as a known limitation if a judge asks.

## 3. Prebuilt spatiotemporal transformer

The architecture family the PS wants (shared backbone, multi-task heads, spatiotemporal attention, low latency vs NWP) is best matched by **Earthformer** (Amazon Science, NeurIPS 2022):
- Paper: https://arxiv.org/abs/2207.05833
- Code: https://github.com/amazon-science/earth-forecasting-transformer
- Built on "Cuboid Attention" — space-time attention that's efficient enough to run on gridded weather data, current SOTA on the SEVIR precipitation-nowcasting benchmark.

**Caveat:** their public checkpoints are pretrained on single-channel radar (SEVIR) and ENSO data, not on IMDAA/INSAT multi-variate grids — so you can't load their weights directly, only reuse the architecture. Two paths:
- **(A) Full Earthformer:** clone their repo, import `CuboidTransformerModel`, retrain from scratch on your fused tensor. Best accuracy, more setup/GPU time.
- **(B) Lightweight custom version (included in this repo, `models/spatiotemporal_transformer.py`):** implements the same axial space-time attention idea in ~150 lines, multi-task heads for thunderstorm/cloudburst/flash-flood, no extra dependencies, verified to run and produce correctly-shaped outputs. Faster to get working within hackathon time; swap in (A) later via the `EarthformerBackboneAdapter` stub if you have GPU budget.

## 4. Pipeline (run in order)

```bash
pip install torch xarray kagglehub cdsapi earthaccess elevation --break-system-packages

# 1. Pull data (defaults to open fallbacks)
python data/fetch_data.py --dataset all --start 2023-06-01 --end 2023-06-10

# 2. Align everything onto one spatiotemporal grid (see preprocessing/align_grid.py)
python -c "
from preprocessing.align_grid import align_all, to_tensor_windows
merged = align_all(thermo_ds, iwv_ds, precip_ds, dem_da)   # load your fetched files into these first
samples = to_tensor_windows(merged, context_len=6, horizon_len=6)
"

# 3. Train the multi-task model
python train.py

# 4. Run inference -> risk maps -> alerts
python infer.py
```

## 5. What's stubbed vs. what's tested

- ✅ Model forward pass is tested (`models/spatiotemporal_transformer.py`) — confirmed it outputs 3 correctly-shaped risk maps from a dummy fused tensor.
- ⚠️ `data/fetch_data.py` and `preprocessing/align_grid.py` are written against real APIs (kagglehub, cdsapi, earthaccess, elevation) but need your own free API keys (CDS, Earthdata) to actually pull data — register these today, they're instant/free unlike the India-specific portals.
- ⚠️ Labels in `train.py` are physically-motivated heuristics, not ground truth — good enough for a working demo; call out this limitation explicitly in your pitch, and mention the IMD-bulletin labeling plan as future work.
