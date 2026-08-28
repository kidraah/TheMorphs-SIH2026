# INSAT format check — recorded baseline

satpy's `insat3d_img_l1b_h5` reader was written against older sample files.
If MOSDAC's product has drifted, the reader **fails silently** — plausible
arrays with wrong values — rather than raising. So a PASS must always be
attributed to a specific product version.

## Baseline: 2019 V01R00 — PASSES

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

## Open: current product version — NOT yet verified

A 2019 pass says nothing about MOSDAC's output today. INSAT-3DS became
operational Feb 2024 and may carry a different `datasetId`, `Software_Version`
and possibly geometry.

To close it: fetch `configs/mosdac_test.json`, run

```bash
.venv/bin/python -c "from nowcast_data.insat import ingest_scan; a,c = ingest_scan('PATH.h5', strict=False); print(c.report())"
```

and diff the printed identity block against the table above. Differences in
`Software_Version` or `Product_Type` are the ones to worry about.
