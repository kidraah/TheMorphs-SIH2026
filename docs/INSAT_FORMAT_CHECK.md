# INSAT format check — recorded baseline

satpy's `insat3d_img_l1b_h5` reader was written against older sample files.
If MOSDAC's product has drifted, the reader **fails silently** — plausible
arrays with wrong values — rather than raising. So a PASS must always be
attributed to a specific product version.

## What is proven, and what is not

| satellite | product | satpy reader | our `ingest_scan` | status |
|---|---|---|---|---|
| INSAT-3D | `3DIMG_L1B_STD` | PASS (2019) | PASS | In-Active since 2024-06-18 |
| INSAT-3DR | `3RIMG_L1B_STD` | **PASS** (2025-08-01) | **PASS** | Active — the workhorse |
| INSAT-3DS | `3SIMG_L1B_STD` | **FAILS** | **PASS** (native fallback) | Active — current operational |

## ARCHIVE CONFIGS ARE CLEARED

`3RIMG_L1B_STD` passes both satpy and our full check on a real 2025 scan.
All 24 monsoon archive configs and every hindcast event from 2016 onward are
cleared to run.

**Caveat, per the run:** the gate files are dated **2025-08-01**, not 2026.
Both are inside coverage so the gate is valid, but a format change in the
last year would not be caught. Re-verify with a recent scan before the
operational deployment claim.

## satpy FAILS on INSAT-3DS — cause found, worked around

```
KeyError: "No variable named 'Longitude_WV'"
```

Structural, not a corruption. On 3D and 3DR the water-vapour channel is
half-resolution (1408×1402, 8 km) and carries its own coarse geolocation
arrays `Latitude_WV` / `Longitude_WV`. **On 3DS the WV channel is full 4 km
(2816×2805)**, shares the main `Latitude`/`Longitude`, and the `*_WV` arrays
do not exist. satpy 0.60 assumes they do.

`nowcast_data.insat.read_scan_native` decodes directly from the HDF5 via the
count→temperature LUT and resolves geolocation per channel, so 3DS works.
`ingest_scan` uses satpy first and falls back automatically, recording which
reader ran.

**This changes a design assumption.** The whole per-channel resolution
argument — degrade SEVIR water vapour to 8 km because that is what INSAT
resolves — holds for 3D and 3DR but **not for 3DS, where WV is 4 km.** A
model pretrained on 8 km WV and deployed on 3DS would be discarding real
resolution; one trained on 3DS WV and run on 3DR would expect detail that is
not there. Training on 3DR alone (which spans the entire archive) sidesteps
this, and it is another reason to prefer that.

## INSAT-3DS writes a garbage solar-elevation attribute

`Sun_Elevation(Degrees) = 7.68e-76` on the 2025-08-01 2330Z scan. It is
denormal-small rather than out of range, so a naive `float()` accepts it and
reads as "0 degrees". That gated correctly here only because the scan is at
night anyway; on a daytime scene it would have let the reflective checks run
against unknown illumination.

`solar_elevation()` now rejects values within 1e-6 of zero, and falls back to
the per-pixel `Sun_Elevation` dataset **sampled over the AOI** — not the
whole disk. A geostationary full disk always spans the terminator, so its
median elevation is ~0 by construction: measured at **+0.80°** for the 3DR
scan whose own attribute reads **−17.26°**. A disk-median fallback would have
reported every scan as twilight.

AOI-sampled (central India): 3DR −1.13°, 3DS −4.09° — both correctly pre-dawn.

## Calibration method differs between satellites

| satellite | `Radiometric_Calibration_Type` |
|---|---|
| INSAT-3D (2019) | LAB CALIBRATED |
| INSAT-3DR (2025) | LAB CALIBRATED |
| INSAT-3DS (2025) | **ONLINE CALIBRATED** |

Relevant to whether the three can be mixed in training — see the
cross-satellite section below.

## Baseline: 2019 V01R00 (INSAT-3D) — PASSES

`3DIMG_25AUG2019_2330_L1B_STD_V01R00.h5` (400 MB), satpy 0.60.0, h5netcdf.

```
Satellite_Name               INSAT-3D
Sensor_Id                    IMG
Processing_Level             L1B
Product_Type                 STANDARD(FULL DISK)
Software_Version             1.0
HDF_Product_File_Name        3DIMG_25AUG2019_2330_L1B_STD.h5
Acquisition_Date             25AUG2019
Acquisition_Time_in_GMT      2330
Sun_Elevation(Degrees)       -12.563013
Radiometric_Calibration_Type LAB CALIBRATED
```

Result: **ALL PASS**

| channel | shape | finite | range |
|---|---|---|---|
| TIR1 | 2816 × 2805 | 72.9% | 180.09 – 330.56 K |
| WV | 1408 × 1402 | 73.0% | 179.89 – 277.68 K |

Native geometry confirms the resolution hierarchy the whole pipeline is
built on: TIR 2816×2805 (4 km), WV 1408×1402 (half → 8 km), VIS/SWIR
11264×11220 (4× → 1 km).

Sun elevation −12.6° (05:00 IST), so VIS and SWIR were correctly reported
**N/A** rather than failed. Requires `h5netcdf` installed.

### This forced a correction to our own thresholds

The original check used a textbook 190–320 K window for TIR-1. That would
have **failed this entirely valid file** — full disk reaches the Arabian and
Saharan surface at local noon (>330 K) and overshooting convective tops punch
below 190 K. Widened to 170–345 K (TIR/MIR) and 170–290 K (WV) against
observed data. A sanity check calibrated on a textbook rather than on the
instrument is itself a source of false alarms.

## Gate results, 2025-08-01

### INSAT-3DR — `3RIMG_01AUG2025_2345_L1B_STD_V01R00.h5` (433 MB) — PASS

```
Satellite_Name               INSAT-3DR      Software_Version   1.0
Product_Type                 STANDARD (FULL DISK)
Acquisition_Time_in_GMT      2345           Sun_Elevation      -17.26 deg
Radiometric_Calibration_Type LAB CALIBRATED
Sub-satellite longitude      74.0 E
```

| channel | shape | finite | p1 | p50 | p99.9 | floor saturation |
|---|---|---|---|---|---|---|
| TIR1 | 2816×2805 | 72.9% | 193.4 | 278.3 | 298.7 | 0.339% @ 179.86 K |
| WV | 1408×1402 | 73.1% | 208.7 | 243.2 | 264.2 | 0.000% |

### INSAT-3DS — `3SIMG_01AUG2025_2330_L1B_STD_V01R00.h5` (423 MB) — PASS via native reader

```
Satellite_Name               INSAT-3DS      Software_Version   1.0
Acquisition_Time_in_GMT      2330           Sun_Elevation      7.68e-76 (GARBAGE)
Radiometric_Calibration_Type ONLINE CALIBRATED
Sub-satellite longitude      82.0 E
```

| channel | shape | finite | p1 | p50 | p99.9 | floor saturation |
|---|---|---|---|---|---|---|
| TIR1 | 2816×2805 | 72.8% | 211.8 | 279.8 | 299.1 | 0.002% @ 180.00 K |
| WV | **2816×2805** | 72.6% | 210.2 | 242.2 | 262.3 | 0.000% |

### Sentinel scan — clean on both

Fourth data source checked, and the first with nothing to report. Once the
off-disk fill (count 1023, ~27% of pixels) is masked, neither file has an
isolated value spike; only low-suspicion interior modes from LUT
quantisation. The LUT clamp is present as expected — 3D 180.09 K, 3DR
179.86 K, 3DS 180.00 K, all slightly different — and is correctly reported
as visible-but-passing rather than treated as physics.

### Cross-satellite comparison — INCONCLUSIVE, and that is the honest answer

Compared on the **common India-grid footprint** (787,968 cells), not on raw
full disks: 3DR sits at 74°E and 3DS at 82°E, so their disks cover different
geography and a raw distribution comparison would measure that rather than
calibration.

```
        p1     p5    p10    p25    p50    p75    p90    p95    p99  p99.9
     +13.42  +7.66  +4.74  +2.51  +0.88  +0.62  +0.97  +0.91  +0.82  +0.91
```

The warm half is a clean **+0.90 K offset** (spread 0.97 K) — small, and the
kind of thing per-satellite normalisation handles. The cold tail diverges by
**13.4 K**.

Calibration shifts the *whole* distribution, so a tail-only divergence points
elsewhere: the two scans are 15 minutes apart with convection evolving, and
deep cloud tops are strongly parallax- and view-angle-sensitive across an 8°
difference in sub-satellite longitude. **One scene pair cannot separate those
from a real calibration difference.** Repeat over many coincident pairs,
preferably clear-sky, before deciding.

This does not block anything: 3RIMG alone spans 2016-10-11 to present and
covers every hindcast event, so training need not wait on the answer.

To close it: fetch `configs/mosdac_test.json`, run

```bash
.venv/bin/python -c "from nowcast_data.insat import ingest_scan; a,c = ingest_scan('PATH.h5', strict=False); print(c.report())"
```

and diff the printed identity block against the table above. Differences in
`Software_Version` or `Product_Type` are the ones to worry about.
