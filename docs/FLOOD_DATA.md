# Flood track: what is on disk, what is missing, what to download

## What is actually on disk

**`/Users/evad/MERIT`, 17 GB, two layers:**

| layer | what it is | tiles |
|---|---|---|
| `hnd` | HAND — height above nearest drainage, m | 4 groups, 132 tiles |
| `upa` | upstream drainage area, km² | 4 groups, 132 tiles |

Nothing else. The brief said "HAND, upa, flow direction, flow accumulation,
basins and rivers"; on disk there are **two** layers, not six. Two of those
names are the same thing — **`upa` IS flow accumulation**, expressed as
contributing area rather than a cell count — and flow direction, basins and
rivers are not present in any form. This matters because flow direction is
the layer the whole track routes on.

## Required, missing, blocking  — RESOLVED, both downloaded

`dir` and `elv` are on disk, all eight groups, and the flow-direction gate
has passed on two tiles (LIMITATIONS #8). The reasoning below is kept
because both mistakes are easy to repeat.


**1. MERIT Hydro `dir` — flow direction.** Load-bearing. Everything
downstream of it (accumulation, sub-basins, channel length, routing) is a
function of it. Same registration you already have, same tile grid, ~17 GB
for the eight groups.

**2. MERIT Hydro `elv` — hydrologically adjusted elevation.** Required for
**channel slope**, which Kirpich needs. HAND is not a substitute: it is
height above the *nearest stream*, not elevation, so it cannot give a
downstream gradient. Without `elv`, `channel_slope` is nan and every arrival
time is nan — deliberately, rather than a default slope producing confident
wrong warning times. ~17 GB.

Both are at `http://hydro.iis.u-tokyo.ac.jp/~yamadai/MERIT_Hydro/`, and
`upa`/`hnd` came from there, so the URL pattern is known-good. Take the same
eight groups: `n00e060 n00e090 n30e060 n30e090` for `dir` and `elv`.

**Verification gate before any flood number is believed.** The D8 convention
in `nowcast_flood/flow.py` is asserted from documentation, not verified — no
`dir` tile has ever been read here. A transposed or reversed encoding routes
water confidently in the wrong direction, produces basins, and returns
plausible numbers. So `verify_against_upa()` recomputes accumulation from
`dir` and checks it reproduces MERIT's own `upa`, which is already on disk.
Two independently derived layers agreeing is cheap proof; run it first.
This is [DATA_TRUST_RULE.md](DATA_TRUST_RULE.md) applied to a raster whose
errors are invisible.

## Required for curve number

**3. Land cover — ESA WorldCover 2021 v200.** 10 m, CC-BY 4.0, **no
registration and no throttle**, from `s3://esa-worldcover/v200/2021/map/`
(public bucket) or Zenodo. India is ~30 tiles of 3°×3°, ~25 GB; it
downsamples to the 4 km grid by majority class, so 10 m is far more than
needed and the download is the only cost.

*On Bhuvan LULC:* thematically better for India — it separates kharif/rabi
cropping, which WorldCover collapses into one "cropland" class, and cropping
season changes CN materially. But it is throttled, needs registration, and
is tiled on the AWiFS scheme. **Recommendation: start with WorldCover**, and
treat Bhuvan as a later refinement rather than a blocker. The CN mapping in
`curve_number.py` is a dict keyed by class code precisely so a Bhuvan legend
can be swapped in without touching the maths.

**4. Soil — HYSOGs250m.** ORNL DAAC, **CC0**, 250 m, global. This is
*hydrologic soil groups A/B/C/D directly*, built for curve-number work, so
there is no derivation from sand/clay fractions and no judgment call to get
wrong. Prefer it over SoilGrids for exactly that reason. Codes 1–4 are
A–D; 11–14 are the drained variants, which `curve_number()` maps back to
their parent group (conservative, and stated — dropping them would blank
much of the Indo-Gangetic plain).

## Required for exposure

**5. Population — GHS-POP R2023A, 100 m.** JRC, free, no registration.
Without it, exposure is **area only**, and ranking basins by flooded area
puts empty Himalayan headwaters above Mumbai. `ExposureResult.report()` says
so in every run that lacks it rather than quietly ranking on area.

## Exact tile lists for the remaining three

Computed from the analysis grid's own footprint, not a hand-drawn box:
lat 5.796-39.491, lon 62.283-101.717. The LAEA footprint is a **rotated
square** in lat/lon, so its lat/lon bounding box over-counts the corners --
182 WorldCover tiles by bbox versus **151** by true intersection. The full
lists are in `configs/flood_tiles.json`.

### ESA WorldCover 2021 v200 -- 151 tiles

Named by lower-left corner on a 3x3 degree grid:

    ESA_WorldCover_10m_2021_v200_{N|S}yy{E|W}xxx_Map.tif

spanning `N03E066` to `N39E099`. From the public bucket
`s3://esa-worldcover/v200/2021/map/` (no credentials) or Zenodo.

**Expect 404s and do not treat them as errors.** WorldCover publishes land
tiles only, so the ocean-only tiles in this list -- a good share of the
Arabian Sea and Bay of Bengal corners -- simply do not exist. Budget ~90 of
the 151 and ~65 GB rather than the full ~113 GB.

**Cheaper alternative worth considering first:** the tiles are COGs, so the
4 km curve-number grid can be built by reading overview levels directly over
`/vsis3/` without downloading anything. We downsample 10 m to 4 km by
majority class, which throws away 99.99% of the pixels we would have paid to
transfer.

### HYSOGs250m -- 1 file

`HYSOGs250m.tif`, a single global GeoTIFF, ~1.6 GB, from ORNL DAAC. No
tiling and no subsetting needed.

### GHS-POP R2023A 100 m -- 22 tiles

Mollweide (ESRI:54009), 1,000,000 m square tiles, rows R5-R9 and columns
C24-C28:

    R5_C24, R5_C25, R5_C26, R5_C27, R5_C28, R6_C24, R6_C25, R6_C26, R6_C27, R6_C28, R7_C25, R7_C26, R7_C27, R7_C28, R8_C25, R8_C26, R8_C27, R8_C28, R9_C25, R9_C26, R9_C27, R9_C28

Full names: `GHS_POP_E2025_GLOBE_R2023A_54009_100_V1_0_R{r}_C{c}.zip`.
E2025 is the epoch nearest to now; E2020 is the last census-anchored one if
you would rather not use a projection.

## Priority

| # | layer | blocks | size |
|---|---|---|---|
| ~~1~~ | MERIT `dir` | — | **done, gate passed** |
| ~~2~~ | MERIT `elv` | — | **done** |
| 1 | HYSOGs250m | runoff | ~1.6 GB, 1 file |
| 2 | ESA WorldCover | runoff | ~65 GB of ~90 real tiles |
| 3 | GHS-POP | exposure ranking | 22 tiles |

3 and 4 together turn rain into runoff; until they land, `route()` takes a
curve-number raster directly, so the chain can be exercised with a constant
CN. GHS-POP only affects *ranking*, so it is genuinely last.

**One more input, not on the original list: antecedent rainfall.** AMC moves
runoff by 15.3x on identical rain, so it is derived from the 5-day
antecedent IMERG accumulation rather than assumed — which makes it a
**cache-config channel** that must be frozen before bulk ingest. See
LIMITATIONS #11. No new download: it comes from IMERG, which is already
being pulled.

None of these URLs have been fetched from this session. Per
[DATA_TRUST_RULE.md](DATA_TRUST_RULE.md), an HTTP 200 is not evidence a file
exists — range-check the first bytes of the first tile of each product
before trusting the set, which is how the MERIT relocation page was caught.

## Licences

| product | licence | note |
|---|---|---|
| MERIT Hydro | CC-BY-NC 4.0 | **non-commercial**; already recorded in DATA_DECISIONS.md |
| ESA WorldCover | CC-BY 4.0 | attribution |
| HYSOGs250m | CC0 | public domain |
| GHS-POP | CC-BY 4.0 | attribution |

MERIT's NC clause is the binding one for the whole flood track, since `dir`,
`elv`, `hnd` and `upa` all come from it.
