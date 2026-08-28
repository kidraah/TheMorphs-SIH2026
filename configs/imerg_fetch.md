# IMERG fetch — GES DISC

Labels for the Indian heads. See `docs/LABELS.md` for why IMERG and not
INSAT-derived IMSRA/HEM.

## One-time setup

1. Earthdata Login: https://urs.earthdata.nasa.gov
2. **Authorise the archive** — this step is easy to miss and produces a
   confusing 401 later: Earthdata profile → Applications → Authorized Apps →
   approve **NASA GESDISC DATA ARCHIVE**.
3. Put the credentials in the gitignored `.env`:

```
EARTHDATA_USERNAME=...
EARTHDATA_PASSWORD=...
```

`.netrc` is also needed for `wget`-style access:

```
machine urs.earthdata.nasa.gov login YOUR_USER password YOUR_PASS
```
(`chmod 600 ~/.netrc`)

## Product

| | |
|---|---|
| dataset | `GPM_3IMERGHH` version 07 (half-hourly, 0.1°, Final Run) |
| variable | `Grid/precipitation` (V07; was `precipitationCal` in V06 — the reader tries both) |
| cadence | 30 min, matching the INSAT scan cadence |
| granule size | ~25 MB global |

Use the **Final Run** for training labels. The Late/Early runs exist for
real-time operation and have different bias characteristics — do not mix them
into one training set.

## Archive path

```
https://gpm1.gesdisc.eosdis.nasa.gov/data/GPM_L3/GPM_3IMERGHH.07/<YYYY>/<DDD>/
```

`<DDD>` is day-of-year, zero-padded. 25 Aug 2019 → `2019/237/`.

## P0 request — the alignment gate

One granule, to pair with the INSAT scan already on disk:

- date: **2019-08-25**, granule **2330–2400 UTC**
- path: `GPM_3IMERGHH.07/2019/237/`
- bbox: none; the file is global and we subset locally
- size: ~25 MB

Then:

```bash
cd /Users/evad/nowcast && .venv/bin/python -c "
from nowcast_data.imerg import ingest
from nowcast_data.insat import ingest_scan
from nowcast_data.alignment import alignment_offset
rain, _ = ingest('IMERG.HDF5', strict=False)
sat, chk = ingest_scan('3DIMG_25AUG2019_2330_L1B_STD_V01R00.h5', strict=False)
print(chk.report())
print(alignment_offset(sat['TIR1'], rain).report())
"
```

## Bulk

For P5/P6, subset server-side rather than pulling global granules: GES DISC
OPeNDAP accepts a lat/lon range in the request, which cuts each granule from
~25 MB to well under 1 MB for the India box. Pulling global and cropping
locally wastes roughly 30× the bandwidth and disk.
