# Download manifest

Everything the project needs, in priority order. You are the only one who can
authenticate to MOSDAC and GES DISC, so this is written to be fetched once
rather than iterated on.

**Sizes are estimates** except where marked measured. **Disk: ~271 GB free** —
which is the binding constraint on items P5 and P6, see the note there.

Date format `DDMMMYYYY` (e.g. `25AUG2019`) is the form seen in the working
2019 request and in the delivered filename. Times are UTC; IST = UTC + 5:30.

---

## P0 — IMERG for 25 Aug 2019, 23:30 UTC · ~25 MB · **do this first**

| | |
|---|---|
| source | GES DISC, `GPM_3IMERGHH.07` |
| window | 2019-08-25 23:30–00:00 UTC (one half-hourly granule) |
| bbox | none — global file, we subset locally |
| count | 1 |
| unblocks | **the alignment gate, today** |

Cheapest high-value item on the list. We already hold
`3DIMG_25AUG2019_2330_L1B_STD_V01R00.h5`, so one matching IMERG granule lets
`alignment_offset` run immediately and answer whether INSAT and IMERG land in
the same place on our grid. That check gates all Indian training data, and a
systematic offset found now costs nothing while the same offset found after
training costs the training.

Caveat: 23:30 UTC is 05:00 IST, so convection is nocturnal and may be sparse.
If the check reports "no pixels colder than 240 K", it is not a failure —
fetch a daytime pair (P2) and rerun.

Direct: `https://disc.gsfc.nasa.gov/datasets/GPM_3IMERGHH_07/summary` →
Subset/Get Data → date 2019-08-25, or the archive path
`.../GPM_L3/GPM_3IMERGHH.07/2019/237/`  (237 = day-of-year for 25 Aug 2019).

---

## P1 — INSAT format-check scan · ~400 MB · **the gate on everything else**

| | |
|---|---|
| config | `configs/mosdac_test.json` |
| datasetId | **uncertain** — see below |
| window | a recent monsoon afternoon, 0800–0900 UTC (1330–1430 IST) |
| bbox | empty (full disk, deliberately) |
| count | 1 |
| unblocks | every subsequent INSAT download |

**datasetId is the open question.** `3DIMG_L1B_STD` is confirmed working for
2019 and returns a parameter error for 2026. Try in this order and record
which works:

1. `3SIMG_L1B_STD` — INSAT-3DS, operational since Feb 2024, and the MOSDAC
   manual's own example. Most likely for recent dates.
2. `3DRIMG_L1B_STD` — INSAT-3DR.
3. `3DIMG_L1B_STD` — INSAT-3D; may now be archive-only.

Full disk, not a subset: the check must exercise the same product geometry
the reader was written against, and a spatial subset could mask a geometry
change. Daytime so VIS/SWIR carry signal and the reflective checks actually
run instead of being skipped.

What we already know: satpy 0.60 reads a **2019 V01R00** file correctly
(TIR1 180.1–330.6 K, WV 179.9–277.7 K, all checks pass). That says nothing
about the current product. `read_metadata()` prints `Software_Version`,
`Product_Type` and `HDF_Product_File_Name` — diff those against the 2019
values recorded in `docs/INSAT_FORMAT_CHECK.md`.

---

## P2 — Indian event windows for hindcasting · ~50–100 GB · needed for the real claim

Known high-impact events, all inside the INSAT-3D/3DR era. Each needs INSAT
**and** matching IMERG. These are the hindcast validation set — the cases a
judge will ask about by name.

| event | date(s) UTC | why |
|---|---|---|
| Chennai floods | 2015-12-01 → 12-02 | urban flood, well documented |
| Mumbai deluge | 2017-08-29 | 300+ mm in a day |
| Kerala floods | 2018-08-15 → 08-17 | orographic, the DEM case |
| Hyderabad | 2020-10-13 → 10-14 | urban convective |
| Uttarakhand | 2021-10-17 → 10-19 | Himalayan orographic |
| Amarnath cloudburst | 2022-07-08 | true cloudburst, station-scale |
| Bengaluru | 2022-09-05 | urban |
| Himachal | 2023-07-08 → 07-11 | sustained orographic |
| Wayanad landslide | 2024-07-30 | landslide, DEM-coupled |

Per event: 0600–1800 UTC (24 scans/day) with the India bbox.
**Fetch 2–3 events first**, not all nine — enough to build and debug the
hindcast path, then extend.

---

## P3 — DEM and MERIT Hydro · ~5–10 GB · unblocks the flash-flood head

| | |
|---|---|
| source | MERIT Hydro (`hydro.iis.u-tokyo.ac.jp/~yamadai/MERIT_Hydro/`), free registration |
| layers | elevation, flow direction (`dir`), flow accumulation (`upa`), width |
| extent | 6–38 °N, 68–98 °E |
| unblocks | basin delineation, and limitation #3 in `docs/LIMITATIONS.md` |

Static — download once, never again. CartoDEM (Bhoonidhi) is the ISRO
alternative and better for the pitch, but MERIT Hydro ships the
hydrologically-conditioned flow layers already derived, which is the
expensive part to reproduce.

---

## P4 — ERA5 thermodynamics · ~20–40 GB · replaces IMDAA for now

| | |
|---|---|
| source | Copernicus CDS, `reanalysis-era5-pressure-levels` |
| variables | temperature, specific humidity, geopotential, u/v wind |
| levels | 1000, 925, 850, 700, 600, 500, 400, 300, 250, 200 hPa |
| area | `38/68/6/98` (N/W/S/E — CDS order) |
| times | hourly, restricted to the P2 event windows first |
| unblocks | CAPE/CIN, shear and convergence — the instability and lift terms |

Needs no NCMRWF transfer. Do the event windows before any bulk season.

---

## P5 — INSAT training archive · **multi-TB at full disk — read this first**

Bulk INSAT for model training. **Do not start before P1 passes.**

Arithmetic that constrains this: ~400 MB/scan (measured) × 24 scans/day =
**~9.6 GB/day full disk**. One monsoon season (JJAS, 122 days) is **~1.2 TB**.
Three seasons is ~3.5 TB. This machine has **271 GB free**.

So the bbox in `configs/mosdac_bulk.json.template` is mandatory, not
optional, and even then you should expect to need either external storage or
a cloud volume co-located with the GPU. Confirm MOSDAC's bbox corner order
before trusting the subset — a transposed box silently returns the wrong
region.

Sequence: one week → measure actual subset size → extrapolate → decide
storage → then bulk.

---

## P6 — IMERG training archive · ~500 GB global, far less subset

Matching labels for P5. Same storage logic: subset to the India box at fetch
time via OPeNDAP rather than pulling global granules and cropping locally.

---

## Not needed

- **IMDAA / NCMRWF** — deferred, ERA5 covers the thermodynamics.
- **INSAT IMSRA / HEM** — excluded on principle, not convenience: they are
  INSAT-TIR-derived, so training INSAT → IMSRA learns ISRO's retrieval
  algorithm rather than atmospheric physics. See `docs/LABELS.md`.
- **SEVIR `vis` / `lght`** — already skipped; vis is daytime-only and lght
  has a non-gridded layout.
