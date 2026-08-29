# THEMORPHS — Scientific Data Audit Report
## Cloudburst Prediction System | Himalayan Region | August 2019 Disaster Window

> **Protocol**: READ-ONLY / ZERO MODIFICATION. All metadata extracted programmatically via Python (`rasterio`, `h5py`, `netCDF4`, `zipfile`). Zero files altered, renamed, moved, or deleted.

---

## PHASE 1 RESULTS — Programmatic Metadata Extraction

---

### 1.1 CartoDEM — ISRO Cartosat-1 30m Digital Elevation Model

**Location**: `d:\THEMORPHS\cartoDEM\`

| Parameter | Value |
|:---|:---|
| Total ZIP tiles | **47** |
| Latitude rows covered (°N) | 28, 29, 30, 31, 32, 33 |
| Longitude cols covered (°E) | 72, 73, 74, 75, 76, 77, 78, 79, 80, 81 |
| Combined spatial extent | **W=72.00° E=82.00° S=28.00° N=34.00°** |
| Pixel resolution | **0.000278° × 0.000278°** ≈ **30.8 m** |
| Grid per tile | **3600 rows × 3600 cols** (= 1° × 1° tile) |
| Coordinate Reference System | **EPSG:4326 (WGS84 Geographic)** |
| Datum & Ellipsoid | **WGS84** |
| Data type | `float32` |
| NoData fill value | **-32768.0** |
| Internal TIF naming | `P5_PAN_CD_N{LAT}_000_E{LON}_000_DEM_30m.tif` |

**Representative tile (N28_E72)**:
- Elevation range: **38.9 – 186.3 m** (mean: 93.8 m) — This tile is in the foothill/plains zone
- Higher-latitude tiles (N31–N33) will contain the 2000–7000m+ Himalayan terrain

> [!NOTE]
> The dataset covers **W=72°–82°E, S=28°–34°N**, which maps exactly to the target modeling domain for Himachal Pradesh and Uttarakhand. Note the missing tile `N28_E077` (only 9 tiles in row 28 instead of 10), indicating a data gap near 28°N–77°E (western Garhwal foothills).

**Missing Tile Analysis** (from filename enumeration):
- Row N28: E072, E073, E074, E075, E076, **[E077 MISSING]**, E078, E079, E080, E081 → 9 tiles
- Row N29: E072–E080 → 9 tiles (no E081, E082)
- Row N30: E073–E081 → 9 tiles
- Row N31: E074–E079 → 6 tiles (only central Himalaya)
- Row N32: E073–E079 → 7 tiles
- Row N33: E073–E079 → 7 tiles

---

### 1.2 INSAT-3DR L1B STD — Standard Imager (Full-Disk, Multi-Channel)

**Location**: `d:\THEMORPHS\insat_\3RIMG_L1B_STD\2019\{DAY}\`

| Parameter | Value |
|:---|:---|
| Total `.h5` files | **391** |
| Days covered | 15AUG – 25AUG 2019 (11 days) |
| File size per scan | ~435 MB |
| Total raw data | ~170 GB |
| Satellite | INSAT-3DR |
| Subpoint (nadir) | 0°N, 74°E (lon-74 configuration) |
| Processing level | L1B (calibrated radiances) |
| Full-disk swath | Lat: −81.0° to +81.0°, Lon: −7.2° to +155.2° |
| Time reference | Minutes since 2000-01-01 00:00:00 UTC |
| Temporal cadence (nominal) | **30-minute** scans (some days missing ~0400–0600 IST night gap) |

**Key datasets inside each HDF5 file**:

| Dataset | Shape | Dtype | Units | Fill | Physical Range |
|:---|:---|:---|:---|:---|:---|
| `/IMG_WV` | `(1, 1408, 1402)` | uint16 | (raw counts) | 1023 | 801–993 |
| `/IMG_WV_RADIANCE` | `(1024,)` | float32 | mW·cm⁻²·sr⁻¹·μm⁻¹ | 999 | 0–1.24 |
| `/IMG_WV_TEMP` | `(1024,)` | float32 | K | 999 | **179.7–325 K** |
| `/IMG_TIR1` | `(1, 2816, 2805)` | uint16 | (raw counts) | 1023 | — |
| `/IMG_TIR1_RADIANCE` | `(1024,)` | float32 | mW·cm⁻²·sr⁻¹·μm⁻¹ | 999 | 0–1.766 |
| `/IMG_TIR1_TEMP` | `(1024,)` | float32 | K | 999 | 179.9–340.1 K |
| `/IMG_TIR2` | `(1, 2816, 2805)` | uint16 | (raw counts) | 1023 | — |
| `/IMG_TIR2_TEMP` | `(1024,)` | float32 | K | 999 | 179.9–340.1 K |
| `/IMG_MIR` | `(1, 2816, 2805)` | uint16 | (raw counts) | 1023 | — |
| `/IMG_MIR_TEMP` | `(1024,)` | float32 | K | 999 | 179.7–339.8 K |
| `/IMG_VIS` | `(1, 11264, 11220)` | uint16 | (raw counts) | 0 | — |
| `/IMG_VIS_ALBEDO` | `(1024,)` | float32 | % | 999 | 0–100 |
| `/IMG_SWIR` | `(1, 11264, 11220)` | uint16 | (raw counts) | 0 | — |
| `/Latitude` | `(2816, 2805)` | int16 | degrees_north | 32767 | scale=0.01 |
| `/Longitude` | `(2816, 2805)` | int16 | degrees_east | 32767 | scale=0.01 |
| `/Latitude_WV` | `(1408, 1402)` | int16 | degrees_north | 32767 | scale=0.01, range: -80.88°–80.88°N |
| `/Longitude_WV` | `(1408, 1402)` | int16 | degrees_east | 32767 | scale=0.01 |
| `/time` | `(1,)` | float64 | minutes since 2000-01-01 | — | ~10,320,000 |

**Channel Descriptions**:
- **WV (Water Vapor)**: 6.7 µm — Senses mid-to-upper tropospheric moisture (450–300 hPa layer). Grid: 1402×1408 pixels at ~8 km resolution (4× coarser than TIR).
- **TIR1**: 10.8 µm — Window channel, primary cloud top temperature; 2805×2816 at ~4 km.
- **TIR2**: 12.0 µm — Split-window; used for sea-surface temp and thin cirrus detection.
- **MIR**: 3.9 µm — Mid-infrared; fire detection and low cloud discrimination.
- **VIS**: 0.65 µm — Visible; 11220×11264 at ~1 km resolution.
- **SWIR**: 1.625 µm — Short-wave infrared.

**Temporal slot completeness per day**:
| Day | Files | Gap |
|:---|:---|:---|
| 15AUG | 1 | Only slot 2345 UTC (rest not collected or available) |
| 16AUG | 41 | Missing 1815–2115 IST window |
| 17AUG | 41 | Missing 1815–2115 IST window |
| 18AUG | 40 | Missing 1745+, 1815–2115 window |
| 19AUG | 41 | Full — 11 hours + night |
| 20AUG | 40 | 0546 (off-cadence) present; missing 0845 slot |
| 21AUG | 28 | Large gap: 0015–0545 UTC missing |
| 22AUG | 39 | Missing 0115 slot |
| 23AUG | 40 | Missing 1745 |
| 24AUG | 40 | Missing 1745 |
| 25AUG | 40 | Missing 1745 |

> [!WARNING]
> **15AUG has only 1 STD file** (slot 2345 UTC). This is a critical data gap for the start of the disaster window. The 1745 UTC slots are systematically missing across most days, suggesting satellite maneuver or downlink scheduling.

---

### 1.3 INSAT-3DR L2B CTP — Cloud Top Properties

**Location**: `d:\THEMORPHS\insat_\3RIMG_L2B_CTP\2019\{DAY}\`

| Parameter | Value |
|:---|:---|
| Total `.h5` files | **443** |
| Days covered | 15AUG – 25AUG 2019 |
| Temporal cadence | **30-minute** scans |
| CTP grid satellite nadir | 0°N, 82°E (lon-82 configuration) |
| Full-disk bounds | Lat: −81.0° to +81.0°, Lon: +0.84° to +163.16° |
| Processing level | L2B (geophysical product) |

**Key datasets**:

| Dataset | Shape | Dtype | Units | Fill | Physical Range |
|:---|:---|:---|:---|:---|:---|
| `/CTT` | `(1, 313, 312)` | float32 | **K** | -999.0 | **179.9 – 295.6 K** |
| `/CTP` | `(1, 313, 312)` | float32 | **hPa** | -999.0 | 100 – 778.5 hPa |
| `/EFF_EMISS` | `(1, 313, 312)` | float32 | — | -999.0 | 0.01–1.0 |
| `/CSBT_TIR1` | `(1, 325, 325)` | float32 | K | -999.0 | 179.9–300.3 K |
| `/CSBT_WVR` | `(1, 325, 325)` | float32 | K | -999.0 | 179.7–266.2 K |
| `/CLRFR_TIR1` | `(1, 325, 325)` | float32 | — | -999.0 | 0.005–1.0 |
| `/Latitude` | `(313, 312)` | int16 | degrees_north | 31172 | scale=0.01 |
| `/SAT_ZEN` | `(325, 325)` | float32 | degrees | -999.0 | 0.62–89.88° |

**Key observation**: The CTT field (`/CTT`) has a native grid of **313 × 312 pixels** with the full-disk coverage centered on 82°E. The range 179.9–295.6 K corresponds to deep convective cloud tops at ~200–100 hPa up to warm boundary layer clouds.

---

### 1.4 INSAT-3DR L2B HEM — Hydro-Estimator Precipitation

**Location**: `d:\THEMORPHS\insat_\3RIMG_L2B_HEM\2019\{DAY}\`

| Parameter | Value |
|:---|:---|
| Total `.h5` files | **426** |
| Days covered | 15AUG – 25AUG 2019 |
| Temporal cadence | **30-minute** scans |
| Satellite nadir | 0°N, 74°E |
| Full-disk bounds | Lat: −81.0° to +81.0°, Lon: −7.16° to +155.16° |
| Processing level | L2B (geophysical product) |

**Key datasets**:

| Dataset | Shape | Dtype | Units | Fill | Notes |
|:---|:---|:---|:---|:---|:---|
| `/HEM` | `(1, 2816, 2805)` | float32 | **mm/hr** | -999.0 | Same native grid as TIR1 |
| `/Latitude` | `(2816, 2805)` | int16 | degrees_north | 32767 | scale=0.01 |
| `/Longitude` | `(2816, 2805)` | int16 | degrees_east | 32767 | scale=0.01 |

> [!IMPORTANT]
> The HEM rain rate array is on the **full-disk 2816×2805 grid** (~4 km spatial sampling) — matching the TIR1 channel. This makes geometric alignment between HEM and CTT straightforward (both are derived from the same pixel geometry). The fill value `-999.0` will need masking.

---

### 1.5 INSAT-3DR L2G GPI — GOES Precipitation Index (QPE)

**Location**: `d:\THEMORPHS\3RIMG_L2G_GPI\` AND `d:\THEMORPHS\insat_\3RIMG_L2G_GPI\`

| Parameter | Value |
|:---|:---|
| Total `.h5` files | **136** (68 in each location — duplicates) |
| Days covered | 15AUG – 25AUG 2019 |
| Temporal cadence | **3-hourly** (slots: 0015, 0315, 0615, 0915, 1215, 1515, 2115) |
| Satellite nadir | 0°N, 74°E |
| Grid coverage | Lat: −39.5° to +40.5°, Lon: +30.5° to +120.5° |
| Grid resolution | **1° × 1°** (coarse!) |
| Grid size | **81 × 91 pixels** |
| Processing level | L2G (gridded geophysical product) |

**Key datasets**:

| Dataset | Shape | Dtype | Units | Fill | Range |
|:---|:---|:---|:---|:---|:---|
| `/GPI` | `(1, 81, 91)` | float32 | **mm** | -999.0 | 0 – 9 mm |
| `/latitude` | `(81,)` | float64 | degrees_north | — | -39.5 to +40.5 |
| `/longitude` | `(91,)` | float64 | degrees_east | — | 30.5 to 120.5 |

> [!CAUTION]
> GPI is on a **1° × 1° grid** (81×91 points) — far coarser than both HEM (~4 km) and the target 0.04° grid. It must be bilinearly interpolated × 25 to reach the target resolution. Also note: **3-hourly** cadence vs HEM's **30-minute** cadence — GPI provides only ~7 samples/day vs HEM's ~48 samples/day.

---

### 1.6 IMDAA Reanalysis — HGT 300 mb (and many more variables)

**Location**: `d:\THEMORPHS\IMDAA\2de719f5-52b8-4e32-b002-b93a15038dcb.zip`

| Parameter | Value |
|:---|:---|
| Archive size | **4.54 GB** (compressed) |
| Total `.nc` files inside | **7,440** |
| File format | NetCDF4-Classic (CF-1.6 convention) |
| CDO processing history | `cdo -f nc4c -z zip_4 sellonlatbox,30.0,120.0,-15.0,45.0` |
| Spatial domain | **30°E–120°E, 15°S–45°N** (pre-cropped) |
| Grid dimensions | **501 lat × 751 lon** |
| Horizontal resolution | ~0.12° (≈ 12 km NCUM) |
| Temporal resolution | **3-hourly** (00, 03, 06, 09, 12, 15, 18, 21 UTC) |
| Time reference | hours since 2019-08-01 00:00:00 |

**Variables identified in the IMDAA archive** (from filenames):

| Variable Code | Physical Meaning | Pressure Levels |
|:---|:---|:---|
| `HGT` | Geopotential Height | 300, 500, 700, 850, 925, 1000 hPa |
| `UGRD` | U-component of Wind | 300, 500, 700, 850, 925, 1000 hPa |
| (likely more) | Temperature, Humidity, etc. | Multiple levels |

**Sample NC file structure** (`UGRD-700mb_2019080103_ncum_imdaa_reanl_prl_03_700_hpa.nc`):

| Variable | Shape | Dtype | Units | Long Name |
|:---|:---|:---|:---|:---|
| `time` | `(1,)` | float64 | hours since 2019-08-01 | time |
| `lon` | `(751,)` | float64 | degrees_east | longitude |
| `lat` | `(501,)` | float64 | degrees_north | latitude |
| `plev` | `(1,)` | float64 | Pa | pressure |
| `u` | `(1, 1, 501, 751)` | float32 | m s⁻¹ | U component of wind |

> [!NOTE]
> Each NC file contains **a single variable, a single time step, and a single pressure level** — this is a "one-variable-per-file" chunking strategy from CDO. With 7,440 files, if we assume 6 pressure levels × 8 time steps/day × ~31 days (Aug 2019) × N variables ≈ `7440 / (6 × 8 × 31) ≈ 5 variables`. So likely: HGT, UGRD, VGRD, TMP, SPFH or similar.

---

## PHASE 2 — Master Data Manifest Table

| Folder / Product | Variable Name in File | Physical Parameter | File Format | Native Spatial Res | Temporal Res | Date Range | Coordinate Extent | Physical Unit |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `cartoDEM/` | `/Band_1` (TIF band 1) | Terrain Elevation (DEM) | GeoTIFF inside `.zip` | **~30.8 m** (0.000278°) | Static (timeless) | Climatological | 28–34°N, 72–82°E | meters (m) |
| `insat_/3RIMG_L1B_STD/` | `/IMG_WV` + `/IMG_WV_TEMP` | Water Vapor Brightness Temp (6.7µm) | HDF5 | **~8 km** (WV grid 1402×1408) | **30-min** | 15–25 Aug 2019 | ~81°S–81°N, ~7°W–155°E | K (Kelvin) |
| `insat_/3RIMG_L1B_STD/` | `/IMG_TIR1_TEMP` | Thermal Infrared BT (10.8µm) | HDF5 | **~4 km** (TIR grid 2805×2816) | **30-min** | 15–25 Aug 2019 | ~81°S–81°N, ~7°W–155°E | K (Kelvin) |
| `insat_/3RIMG_L2B_CTP/` | `/CTT` | Cloud Top Temperature | HDF5 | **~4 km equivalent** (312×313 full-disk) | **30-min** | 15–25 Aug 2019 | ~81°S–81°N, ~1°–163°E | K (Kelvin) |
| `insat_/3RIMG_L2B_CTP/` | `/CTP` | Cloud Top Pressure | HDF5 | **~4 km equivalent** (312×313) | **30-min** | 15–25 Aug 2019 | ~81°S–81°N, ~1°–163°E | hPa |
| `insat_/3RIMG_L2B_HEM/` | `/HEM` | Hydro-Estimator Rain Rate | HDF5 | **~4 km** (2805×2816) | **30-min** | 15–25 Aug 2019 | ~81°S–81°N, ~7°W–155°E | mm/hr |
| `3RIMG_L2G_GPI/` | `/GPI` | GPI QPE Accumulated Rainfall | HDF5 | **~1°** (81×91 grid) | **3-hourly** | 15–25 Aug 2019 | 39.5°S–40.5°N, 30.5–120.5°E | mm |
| `IMDAA/` (zip) | `HGT` | 300 hPa Geopotential Height | NetCDF4 | **~12 km** (501×751 grid) | **3-hourly** | Aug 2019 | 15°S–45°N, 30–120°E | m² s⁻² (geopotential) |
| `IMDAA/` (zip) | `UGRD` | U-wind component | NetCDF4 | **~12 km** (501×751 grid) | **3-hourly** | Aug 2019 | 15°S–45°N, 30–120°E | m s⁻¹ |

---

## PHASE 2 — Domain Science & Physical Significance

### 2.1 — 300 hPa Geopotential Height (`HGT-300mb` from IMDAA)

**What it measures**: The altitude (in meters) of the 300 hPa pressure surface in the upper troposphere (~9–10 km ASL). Since pressure decreases with altitude, lower geopotential height at 300 hPa means cold, dense air aloft — a **trough** or **cutoff low**.

**Physical mechanism in Himalayan cloudbursts**:

Upper-tropospheric dynamics are the "ignition switch" for explosive convection over the Himalayas. Here's why:

1. **Western Disturbances (WDs)** are eastward-moving extratropical cyclones embedded in the midlatitude westerly jet stream at 200–300 hPa. During the Indian Summer Monsoon (June–September), WDs occasionally interact with the northward-displaced monsoon trough, creating a **trough-monsoon interaction** zone exactly over Himachal Pradesh and Uttarakhand.

2. **Negative geopotential height anomaly at 300 hPa** = Cold air aloft = Conditional Instability. When this cold upper air overlies warm, moist monsoonal surface air (surface θe > 350 K), the **Convective Available Potential Energy (CAPE)** exceeds 2000–4000 J/kg.

3. The forced ascent created by the Himalayan orography (terrain slope up to 30–45°) triggers convective initiation. Once the parcel lifts above the Level of Free Convection (LFC), the upper-level divergence associated with the 300 hPa trough acts as an **exhaust pump**, pulling air upward and allowing convective towers to reach the tropopause (200 hPa, ~200 K cloud tops).

4. **The 300 hPa wind speed** (UGRD/VGRD) also tells you if the jet streak is overhead: a jet entrance region creates upper-level divergence (through the ageostrophic wind), which further enhances upward motion.

**In the ML model**: `HGT-300mb` and `UGRD-300mb` are the synoptic "fingerprint" of favorable large-scale conditions. They answer: *"Is the large-scale atmosphere configured to allow convective organization and intensification?"*

---

### 2.2 — Cloud Top Temperature (`/CTT` from L2B CTP)

**What it measures**: The equivalent blackbody temperature of the cloud top as retrieved from the 10.8 µm TIR channel, corrected for emissivity (`/EFF_EMISS`). Units: Kelvin. Valid range in dataset: **179.9 K to 295.6 K**.

- **295 K** = low, warm cloud or cloud-free pixel (boundary layer cumulus)
- **220–240 K** = mid-tropospheric convection (~500–400 hPa, ~6–8 km)
- **<200 K** = deep convective overshooting tops penetrating the tropopause
- **179.9 K** = theoretical cold limit of INSAT-3DR CTT retrieval (physical boundary condition)

**Why CTT cooling rate is the most critical real-time cloudburst signal**:

1. **Explosive CTT Cooling** — When a convective cell transitions from "ordinary shower" to "cloudburst potential," the cloud top drops by **10–15 K in 30 minutes** (or more). This is directly observable in sequential CTT images.

2. **Overshooting Tops (OTs)**: When updraft cores are so intense they pierce the tropopause (CTT < 195 K), they create gravity waves and produce hail, extreme rainfall, and flash flooding. These are visible as localized cold "bullseyes" in the CTT field.

3. **Anvil Spreading**: Once a deep convective cell matures, CTT shows characteristic anvil cloud spreading downstream. The asymmetry in the anvil shape tells you wind shear direction and storm motion.

4. **Minimum CTT Track** (mesoscale convective signature): As a Mesoscale Convective System (MCS) organizes over a Himalayan valley, the CTT minimum migrates up-slope with the valley circulation. Tracking this migration predicts where the cloudburst will hit.

**Key insight for the ML pipeline**: The CTT gradient (`∂CTT/∂t`) computed from successive 30-minute scans is likely the **highest-weight predictive feature** in your ConvLSTM model.

---

### 2.3 — Water Vapor Brightness Temperature (`/IMG_WV` at 6.7 µm from L1B STD)

**What it measures**: Brightness temperature in the 6.7 µm water vapor absorption band. This band senses emission from **mid-to-upper tropospheric water vapor** (roughly 300–600 hPa layer), not the surface. Units: Kelvin. Valid range: **179.7 – 325 K** (from the lookup table `/IMG_WV_TEMP`).

- **Warm (high WV BT ~280–320 K)**: Dry upper troposphere → the satellite "sees" warmer lower troposphere → moisture is absent aloft → suppressed deep convection
- **Cold (low WV BT ~220–240 K)**: Moist upper troposphere → the satellite "sees" only the cold moisture layer → a **moisture plume** is present at 300–500 hPa

**Physical mechanism in Himalayan cloudbursts**:

1. **Moisture Surge Corridors**: Before a cloudburst event, WV imagery shows "dark lanes" (cold BT = moist air) extending from the Bay of Bengal northward along the Gangetic plain into the mountain valleys. This is the **low-level jet (LLJ)** and mid-tropospheric moisture advection.

2. **Anticyclonic WV Loops**: In the upper troposphere (~300 hPa), a well-established Monsoon Anticyclone (centered near 30°N–90°E) has divergent outflow that circulates moisture. Anomalous cyclonic loops or deviations in this pattern indicate upper-level moisture convergence over the Himalayas.

3. **WV vs. TIR1 Synergy**: By comparing `IMG_WV_TEMP` (upper tropospheric moisture) with `IMG_TIR1_TEMP` (cloud top temperature), you can identify **thin cirrus shields** vs **deep convective towers**:
   - If WV_BT ≈ TIR1_BT → deep convective tower, cloud top is at the WV sensing layer
   - If WV_BT >> TIR1_BT → thin cirrus over deep moist layer

4. **Native resolution**: WV grid is 1402×1408 pixels (8 km effective resolution). For your model, this needs spatial upscaling (bilinear interpolation) to match the 4 km TIR grid or 4 km target grid.

**In the ML model**: WV BT captures the **moisture pre-conditioning** of the atmosphere 6–12 hours before deep convection initiation. It acts as a "fuel gauge" for the convective system.

---

### 2.4 — QPE vs. Extreme Rain Rate: GPI (`/GPI`) vs. HEM (`/HEM`)

These two products are fundamentally different and **complementary** — you need both.

#### GOES Precipitation Index (`/GPI`) — L2G Product
- **Algorithm**: Based on GOES Precipitation Index (Arkin & Meisner 1987). Counts pixel-hours where TIR BT < 235 K and multiplies by an empirical rain rate coefficient (3 mm/hr per K-hour). Essentially a threshold-based technique assuming "cold cloud = rain."
- **Grid**: 1° × 1° (coarse) → **81×91 points**, covering the entire India-Asia sector
- **Value range**: 0–9 mm (accumulated over the 3-hour period)
- **Temporal resolution**: 3-hourly
- **Strengths**: Robust, long-track record, climatologically consistent baseline for typical monsoonal rainfall
- **Critical weakness**: **Saturates at 9 mm per 3 hours (~3 mm/hr equivalent)**. During a cloudburst (100–300 mm/hour localized rain), GPI is **completely blind** — it cannot distinguish a 3 mm/hr shower from a 100 mm/hr flash flood.

#### Hydro-Estimator Method (`/HEM`) — L2B Product
- **Algorithm**: NOAA/NESDIS Hydro-Estimator. Uses a non-linear relationship between cloud top temperature and rain rate, corrected for precipitable water and moisture profile. Specifically designed to capture convective extremes.
- **Grid**: Full-disk 2816×2805 pixels (~4 km native resolution)
- **Value range**: 0 to extreme values (mm/hr) — fill=-999.0
- **Temporal resolution**: 30-minute
- **Strengths**: Can resolve extreme convective rain rates; 8× finer temporal resolution than GPI; 4 km spatial resolution captures sub-10 km cloudburst cells
- **Weakness**: Retrieval errors at extreme zenith angles (>70°) and over complex terrain where beam-filling effects occur

**Why you need both in the ML tensor**:

| Use Case | GPI | HEM |
|:---|:---|:---|
| Background monsoon rainfall | ✅ Reliable | ❌ Noisy |
| Convective cells < 50 mm/hr | ✅ Reasonable | ✅ Good |
| Extreme cloudburst > 100 mm/hr | ❌ Saturated at ~9 mm/3hr | ✅ Only option |
| Spatial resolution | ❌ 1° (111 km) | ✅ ~4 km |
| Temporal resolution | ❌ 3-hourly | ✅ 30-min |
| Label/Target for training | Baseline reference | **Primary extreme target** |

**In the ML pipeline**: GPI provides the "climatological background" rain rate that anchors the model in normal monsoon behavior. HEM provides the **high-temporal, high-spatial resolution signal of convective extremes**. Together, they bracket the full dynamic range of precipitation from drizzle to catastrophic cloudbursts.

---

### 2.5 — Topography (`CartoDEM`) — The Silent Trigger

**What it measures**: Absolute terrain elevation in meters above WGS84 ellipsoid at 30m resolution (Cartosat-1 DEM, P5 stereo photogrammetry).

**Physical mechanism — why topography is the primary cloudburst localizer**:

1. **Orographic Forced Ascent**: When moisture-laden monsoonal winds (southwest flow at 850 hPa, ~15–20 m/s) encounter a steep mountain ridge, the air is **forced upward** along the terrain slope. The vertical velocity ω induced by orography is:
   ```
   ω_orographic = -ρg · V⃗ · ∇h
   ```
   where `V⃗` is the horizontal wind vector and `∇h` is the terrain gradient. In Himalayan valleys with slopes of 30–45°, this can generate **vertical velocities of 2–5 m/s**, orders of magnitude above synoptic-scale lifting (~0.01 m/s).

2. **Valley Channeling and Convergence Zones**: Valleys like the Beas, Sutlej, Alaknanda, and Mandakini act as **moisture funnels**. Southwest monsoon winds are channeled up-valley, accelerating due to the pressure gradient and converging at valley bends or where two valleys merge. This convergence adds to the orographic lifting.

3. **Terrain-locked Convective Cells**: Unlike continental MCSs that drift with the steering wind, Himalayan cloudburst cells are **topographically anchored**. The DEM tells the model exactly where the ridgelines, valley axes, and slope-break zones are, making topography the **highest spatial information content feature** in the model.

4. **Derived feature — Terrain Slope**: The slope angle `θ = arctan(|∇h|)` computed from DEM gradients is even more directly relevant than elevation itself:
   - Steep slopes (>20°) → extreme orographic lifting
   - Slope aspect → controls which wind directions trigger ascent
   - Curvature → identifies ridgelines (divergent flow) vs. valleys (convergent flow)

5. **Shadow zones**: On the leeward side of ridges, descending air creates dry, cloud-free zones (föhn effect). The DEM encodes these leeward regions, which will have **zero precipitation** regardless of synoptic conditions — a powerful negative training signal.

**In the ML model**: CartoDEM is a **static channel** (does not change over time). It provides the spatial "grammar" that explains why convection organizes where it does. Including both elevation (absolute height) and slope (gradient) as separate channels allows the ConvLSTM to learn orographic lifting patterns.

---

## PHASE 3 — ML Pipeline Blueprint

### 3.1 — Target Grid Definition

```
Spatial Domain:  N=34.0°, S=27.0°, W=74.0°, E=82.0°
Grid Resolution: Δlat = Δlon = 0.04°  (~4.4 km at 30°N)
Grid Dimensions: H = (34.0 - 27.0) / 0.04 = 175 rows
                 W = (82.0 - 74.0) / 0.04 = 200 cols
Total pixels:    175 × 200 = 35,000
```

> [!IMPORTANT]
> The CartoDEM tiles cover 28–34°N and 72–82°E. The southern extent of the target domain (27°N) is **below the southernmost CartoDEM tiles (28°N)**. Either expand tile acquisition to include 27°N row, or clip the target domain to 28°N.

### 3.2 — Temporal Alignment Strategy

| Source | Native Cadence | Strategy |
|:---|:---|:---|
| INSAT L1B STD (WV, TIR) | 30-min | **Anchor** — defines the temporal spine |
| INSAT L2B CTP | 30-min | Direct 1:1 match |
| INSAT L2B HEM | 30-min | Direct 1:1 match |
| INSAT L2G GPI | 3-hourly | Forward-fill or linear interpolation × 6 to 30-min slots |
| IMDAA (HGT, UGRD, etc.) | 3-hourly | Linear interpolation × 6 to 30-min slots |
| CartoDEM | Static | Broadcast across all time steps |

**Unified timeline**: `t = [2019-08-15 23:45 UTC, 2019-08-16 00:15 UTC, ..., 2019-08-25 23:45 UTC]`
- Total 30-min slots: 11 days × 48 slots = **528 time steps**
- Usable (accounting for STD gaps): ~**391 time steps** with full STD coverage

### 3.3 — Data Preprocessing Strategy

#### A. CartoDEM — DEM Mosaicking and Downsampling
```python
# Step 1: Open all 47 ZIPs in memory (do not extract to disk)
# Step 2: Use rasterio.merge.merge() on in-memory MemoryFile objects
# Step 3: Clip to target bbox [27-34°N, 74-82°E]
# Step 4: Resample from 30m (~0.000278°) to 0.04° using bilinear resampling
# Step 5: Derive slope channel using np.gradient on resampled DEM
```

#### B. INSAT Products — Regridding to Target Grid
```python
# For L2B CTP/HEM (native irregular full-disk geometry):
# Step 1: Read /Latitude and /Longitude arrays (apply scale_factor=0.01)
# Step 2: Mask fill values (32767 for lat/lon, -999 for data)
# Step 3: Use scipy.interpolate.griddata (nearest/linear) to regrid
#          from irregular source (lat_2d, lon_2d) → regular target grid
# Step 4: Clip to target bbox

# For L2G GPI (already on 1°×1° lat-lon grid):
# Simple bilinear interpolation to 0.04° using scipy.RegularGridInterpolator
```

#### C. IMDAA — Reading from ZIP without Extraction
```python
# Open ZIP → read NC bytes → write to tempfile → open with netCDF4
# Pattern: {VAR}-{LEVEL}_{DATE}{HH}_ncum_imdaa_reanl_prl_{HH}_{LEVEL}_hpa.nc
# After reading, interpolate from 0.12° (~12 km) to 0.04° target grid
```

#### D. Missing Data Handling
- **Short gaps (1 slot)**: Linear temporal interpolation
- **Long gaps (>3 consecutive slots, e.g., 21 Aug 0015-0545)**: Flag as `NaN` mask; exclude these windows from training sequences
- **HEM fill (-999)**: Replace with 0 mm/hr (no rain at fill pixels — boundary regions)
- **CTT fill (-999)**: Replace with `NaN` → propagate as invalid mask

#### E. Normalization
| Channel | Strategy | Reason |
|:---|:---|:---|
| Elevation (DEM) | Min-max [0, 1] using domain max (~7500 m) | Bounded, no outliers |
| Slope | Min-max [0, 1] using 90° max | Bounded |
| CTT | Z-score (μ=245 K, σ=25 K) | Near-Gaussian distribution |
| WV BT | Z-score (μ=245 K, σ=30 K) | Near-Gaussian |
| HEM rain rate | Log₁₀(x+1) then min-max | Heavy-tailed distribution; log compresses extremes |
| GPI | Min-max [0, 1] using 9 mm max | Already bounded |
| HGT-300mb | Z-score (μ=9000 m, σ=200 m) | Small variance around mean |

### 3.4 — PyTorch Tensor Specification

#### 5D Input Tensor
```
Shape: (B, T_in, C, H, W)

Where:
  B     = batch size (e.g., 4–16)
  T_in  = input sequence length (e.g., 8 steps = 4 hours of 30-min scans)
  C     = 7 channels:
            C[0] = Elevation (DEM)          — static, replicated T times
            C[1] = Slope (∇DEM)             — static, replicated T times
            C[2] = CTT (Cloud Top Temp)     — dynamic, 30-min
            C[3] = WV BT (6.7 µm)          — dynamic, 30-min
            C[4] = GPI QPE                  — dynamic, 3-hourly → interpolated
            C[5] = HEM Rain Rate            — dynamic, 30-min
            C[6] = HGT-300mb               — dynamic, 3-hourly → interpolated
  H     = 175  (latitudinal pixels, 0.04° spacing, 27-34°N)
  W     = 200  (longitudinal pixels, 0.04° spacing, 74-82°E)

Final tensor shape example (B=8, T_in=8):
  Input:  (8, 8, 7, 175, 200)   → ~490 MB float32 per batch
  Target: (8, 1, 2, 175, 200)   → HEM rain rate + binary cloudburst mask at T+1
```

#### Model Target (Label) Definition
```
Binary Cloudburst Label: HEM[t+1] > 50 mm/hr  (pixel-wise)
Regression Target:       HEM[t+1] (continuous rain rate, log-transformed)
```

---

## Critical Observations & Data Quality Flags

> [!WARNING]
> **15 August STD coverage**: Only 1 file available (slot 2345 UTC). The entire daylight period of 15 Aug is missing from the L1B STD archive. CTP and HEM however have full 30-minute coverage starting 0015 UTC on 15 Aug. This means the first valid input sequence with complete STD data begins on **16 Aug 0015 UTC**.

> [!CAUTION]
> **GPI duplicate data**: The GPI `.h5` files exist in **two identical locations** (`3RIMG_L2G_GPI/` at root level and `insat_/3RIMG_L2G_GPI/`). 136 total files = 68 unique files × 2. Deduplicate before building the dataset.

> [!NOTE]
> **IMDAA temporal coverage**: The IMDAA archive starts from `2019-08-01` (filename: `2019080103`). This gives ~25 days of pre-event context for the 15–25 Aug disaster window — valuable for understanding the synoptic setup evolving through early August.

> [!NOTE]
> **IMDAA variable richness**: Beyond HGT-300mb, the ZIP contains `UGRD` at 300, 500, 700, 850, 925, 1000 hPa — likely also VGRD, TMP, SPFH. These additional variables (especially wind shear = UGRD_300-UGRD_850, and specific humidity at 850 hPa) could significantly improve model skill and deserve consideration as additional channels.

---

*Report generated from programmatic READ-ONLY inspection of workspace `d:\THEMORPHS`. Zero source files modified.*
