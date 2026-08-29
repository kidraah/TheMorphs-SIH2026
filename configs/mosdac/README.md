# MOSDAC download configs

Numbered by priority. Credentials are blank in every file — the download
client reads them from your MOSDAC SSO account; keep real values only in the
gitignored `.env`.

Config shape follows the **verified-working 2023 config**, which the
[API manual](https://mosdac.gov.in/downloadapi-manual) also confirms:
`user_credentials` / `search_parameters` / `download_settings`,
`startTime`+`endTime` as `YYYY-MM-DD`, `boundingBox` as
`minLon,minLat,maxLon,maxLat` (longitude first).

## Why the 2026 request failed

Catalog-verified on 2026-08-29 from `mosdac.gov.in/catalog/satellite.php`:

| datasetId | coverage | status |
|---|---|---|
| `3DIMG_L1B_STD` | 2013-10-01 → **2024-06-18** | **In-Active** |
| `3RIMG_L1B_STD` | 2016-10-11 → present | Active |
| `3SIMG_L1B_STD` | 2024-05-17 → present | Active |

**INSAT-3D stopped delivering on 2024-06-18.** `3DIMG_L1B_STD` with 2026
dates is a valid dataset id outside its coverage window — the parameter error
is the date range, not the syntax. Your 2023 config worked because 2023 is
inside 3D's window.

Note the prefix: INSAT-3DR is **`3RIMG_`**, not `3DRIMG_` as I guessed
earlier. INSAT-3DS is `3SIMG_`.

## Run order

| # | file | what it unblocks |
|---|---|---|
| 00 | `00_verify_3DS_current.json` | **the gate.** Current operational satellite, 1 scan, full disk |
| 01 | `01_verify_3DR_current.json` | **also a gate.** 3DR is a *different satellite* — a reader proven on 3DS is not proven on 3DR, and 3DR is the workhorse for every event from 2016 on |
| 02 | `02_verify_3D_archive.json` | archive retrieval, for the one pre-2016 case |
| 03–14 | `NN_event_*.json` | hindcast validation — the named cases |
| 15–38 | `NN_archive_*.json` | Indian fine-tuning set, monsoon 2023–2025 |

After 00 and 01:

```bash
cd /Users/evad/nowcast && .venv/bin/python -c "from nowcast_data.insat import ingest_scan; a,c = ingest_scan('PATH.h5', strict=False); print(c.report())"
```

Diff the printed identity block against `docs/INSAT_FORMAT_CHECK.md`.

## Which products, and which we deliberately skip

**Take — Imager L1B only:**

- `3RIMG_L1B_STD` — 4 km, half-hourly, 2016-10-11 → present. The workhorse.
  One satellite spans every hindcast event from 2017 and all three monsoon
  seasons, which avoids a sensor discontinuity mid-training.
- `3SIMG_L1B_STD` — 4 km, half-hourly, 2024-05-17 → present. Current
  operational; what a deployed system would actually consume.
- `3DIMG_L1B_STD` — archive only, for Chennai 2015.

**Skip — L2 rain retrievals (`_L2G_IMR` IMSRA, `_L2B_HEM`):** not a
bandwidth decision. Both are derived from INSAT TIR, so training
INSAT → IMSRA teaches ISRO's retrieval algorithm rather than atmospheric
physics. See `docs/LABELS.md`. IMERG stays the label source.

**Skip — other L2B/L2C (`CTP`, `CMK`, `OLR`, `UTH`, `LST`, `FOG`):** derived
from the same radiances we already ingest, so they add no independent
information — only a dependency on someone else's retrieval.

**Defer — AMV / wind (`3SIMG_L2P_AMV`, `3RIMG_L2P_IRW/WVW/VSW/MRW`):** these
*do* add something the imagery lacks — motion vectors, and a shear proxy from
the IR/WV level difference, which maps onto the kinematics term. But they are
sparse vector products, not grids, so they need their own ingest path. Worth
revisiting once the imagery pipeline is validated.

**Defer — Sounder (`3RSND_L2B_SA1`, 10 km, 7×/day):** temperature and
humidity profiles, i.e. CAPE/CIN from INSAT's own instrument. Attractive in
principle, but ERA5 gives hourly profiles on a consistent global grid where
the sounder gives 7 per day at 10 km. ERA5 first; revisit only if it proves
insufficient.

## Volume and the count ceiling

Your 2023 config used `"count": ""` successfully over a 10-day window, so
empty appears to mean unlimited — that empirical result beats the manual's
"Max value: 100". These configs still chunk to **≤2-day windows** (≈96
granules at half-hourly cadence), so they stay under 100 either way.

Full disk is ~400 MB/scan (measured). All event and archive configs therefore
set `boundingBox` to the India analysis grid; that is mandatory rather than
an optimisation, since one unsubsetted monsoon season is ~1.2 TB against
~271 GB free locally.
