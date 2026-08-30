# Before bulk ingest: what must be settled

Ingest transfers **~3.4 TB** and discards the raw files, because MOSDAC does
not subset server-side. Anything missing from the frozen config is therefore
not a small fix — it is a 3.4 TB re-download. The enumeration lives in
`nowcast_train/insat_cache.py` as a hashed dataclass rather than in this
document, so a mismatch is refused rather than remembered.

Current default fingerprint: **`86fae373df9155e7`** (was `a8d4085edddad2de` before Range support was verified; see LIMITATIONS 14) (asserted in the tests;
changing it is a re-ingest and must be deliberate).

## Ordering — confirmed

```
alignment gate  ->  freeze config  ->  ingest events  ->  archive
```

This is right, and the gate genuinely blocks: it measures the parallax
offset between cloud top and ground rain, and that offset is baked into the
labels at ingest. Ingesting first would produce a cache whose satellite and
label fields are misregistered by a distance nobody measured, and no
downstream number would look wrong.

**The gate still needs six `3RIMG_L1B_STD` scans** for 2025-08-01 at
`2245, 2145, 2045, 1845, 1745, 1245` (`configs/mosdac/00b_alignment_gate_scenes.json`).
All matching IMERG granules are already downloaded. Two scenes are on disk;
the gate is INCONCLUSIVE at two and needs five to regress dy on tan(zenith).

## Settled by measurement this round

| item | status |
|---|---|
| per-scan size | **448 MB measured**, not 43 MB estimated — 10.4x |
| does boundingBox subset? | **No.** Delivered files are full Earth disk |
| archive transfer | **3.4 TB**, unavoidable |
| archive *storage* | **117 GB** decoded, because raw is discarded |
| channel set | **all six**, not TIR1+WV — see below |
| WorldCover access | COG overview 1/4 verified; no bulk download |

## The channel decision — the thing most likely to force a re-ingest

Inside one real 448 MB scan, by stored bytes:

| dataset | MB | % | used? |
|---|---|---|---|
| `Longitude_VIS` | 139.1 | 30.8% | no |
| `Latitude_VIS` | 87.9 | 19.5% | no |
| `IMG_SWIR` | 86.4 | 19.2% | no |
| `IMG_VIS` | 80.5 | 17.8% | no |
| everything else | 57.3 | 12.7% | partly |

**The two channels the model reads are 19.1 MB — 4.2% of the file.**

Every VIS and SWIR byte crosses the wire whether or not it is kept. On the
4 km grid a channel costs ~1.6 MB/scan, so keeping all six costs **~47 GB**
more storage and removes any possibility of a 3.4 TB re-download to recover
one. The cost is asymmetric by a factor of ~70, so: **extract everything.**
The model can ignore a channel; it cannot invent one.

VIS/SWIR being day-only is a modelling problem, not a caching one —
`solar_elevation` is cached beside them so the loader gates on illumination
rather than inferring it from the timestamp.

## Also in the freeze, and easy to miss

1. **AMC / antecedent channel** — as you said. Additionally: it needs **5
   days of IMERG *before* the first scan of every event**, because the window
   excludes the current day. Without that lead-in the antecedent channel is
   nan at the start of every event, which is the start of the storm. This is
   an ingest-*window* decision, not only a channel decision, and the config
   refuses a lead-in shorter than the window.
2. **Per-scan metadata that cannot be recovered later**: sub-satellite
   longitude (74E vs 82E — drives parallax, and differs by satellite),
   `solar_elevation` from the *dataset* not the attribute (3DS writes
   denormal garbage), satellite id, which reader was used (satpy vs the
   native LUT fallback), and the physics-check result.
3. **Static-source versions** — MERIT, WorldCover, HYSOGs, GHS-POP. These
   are constant in time but not constant across releases, and a cache built
   against one is not comparable to a cache built against another.
4. **Store the finite mask, not just NaN-filled arrays.** Recovering "was
   this cell missing or is it genuinely zero" after the fact is impossible.
5. **Cache per scan, not per sequence.** Already true of the SEVIR cache and
   carried over: context/horizon lengths then change without a rebuild.
6. **uint16 scaled, not float16.** float16 has ~3 significant decimal digits,
   which at 300 K is a quantisation of ~0.25 K — comparable to the cloud-top
   cooling rates the model keys on.

## Still open, and *not* blocking the freeze

* Kirpich/Watt-Chow/Giandotti are unvalidated against observed arrival times
  (LIMITATIONS #9). Affects reported lead time, not cached bytes.
* The ~0.3% MERIT cell-area offset (LIMITATIONS #8). A known constant.
* FAR ceiling 0.80 is a placeholder (LIMITATIONS #6). Scoring, not caching.
* Flood risk reference discharge is a placeholder (LIMITATIONS #10).

None of these change a cached byte, so none of them should hold up ingest.

## WorldCover via COG overviews — verified, with a correction

The path works with **no credentials and no requester-pays**:

```python
os.environ["AWS_NO_SIGN_REQUEST"] = "YES"
rasterio.open("s3://esa-worldcover/v200/2021/map/"
              "ESA_WorldCover_10m_2021_v200_N21E078_Map.tif")
# 36000x36000 uint8, overviews [2, 4, 8, 16, 32, 64]
```

**Do not read the coarsest overview.** WorldCover's overviews are
nearest-subsampled, not mode-aggregated, so at 1/64 a pixel is one sample
rather than a majority: measured **87.4% agreement** with the true majority,
i.e. one cell in eight gets the wrong land-cover class — and the errors are
biased, since subsampling under-represents fragmented classes like built-up
and water.

Measured agreement with the full-resolution 4 km majority:

| overview | resolution | agrees | tile size |
|---|---|---|---|
| 1/2 | 20 m | 100.00% | 324 MB |
| **1/4** | **40 m** | **100.00%** | **81 MB** |
| 1/8 | 80 m | 99.31% | 20 MB |
| 1/64 | 640 m | 87.4% (direct read) | 0.3 MB |

**Read at 1/4 and compute the majority into 4 km cells yourself.** ~81 MB
per tile, ~7.3 GB for the ~90 land tiles, nothing stored, versus 65 GB of
downloads.

The `average` failure mode is worth naming because it is this project's
usual shape: averaging tree (10) and grass (30) yields shrubland (20) — a
real class, a plausible map, entirely fabricated. WorldCover's overviews are
*not* average-built (every code returned is a valid legend code, checked),
so the risk here is the subsampling bias above, not invented classes.
