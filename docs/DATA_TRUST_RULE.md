# The standing rule for new data sources

> **Assume every new data source contains at least one value that decodes to
> something plausible and wrong. Do not trust the source until the sentinel
> scan and a physics-convention check have both been run against it.**

This is not a precaution against a hypothetical. It is the generalisation of
**ten** separate instances found on this project, by nine different
accidents. None of them raised. All of them produced something a reasonable
person would accept.

The rule started as a rule about data. Instance 9 shows it is not: the same
shape appeared in the **evaluation code**, where it is worse, because a
scoring bug flatters the model rather than degrading it.

## The first five

| # | source | the value | decoded as | why it was dangerous |
|---|---|---|---|---|
| 1 | SEVIR VIL | byte `255` | 81.33 kg/m² | top of the entire range — a missing event became maximum-intensity storm **everywhere**, labelled extreme rain |
| 2 | SEVIR / INSAT IR | int16 min | −327.68 °C | the coldest possible cloud top — the exact signal the CTT drop rate keys on |
| 3 | INSAT L1B | count→K LUT clamp | 180.09 K | appeared in **two channels with different physics**, which is what gave it away |
| 4 | INSAT-3DS | `Sun_Elevation` = `7.68e-76` | "0 degrees" | denormal, not out of range, so `float()` accepted it; gated a night scan correctly *by luck* |
| 5 | ERA5 | CIN = `NaN` | "no inhibition" if filled with 0 | **sign-inverted**: the truth is "no convective parcel exists", i.e. convection impossible. Filling with the obvious default says maximally favourable |

Note the shape they share. Not one was a corruption, a crash, or an obviously
absurd number. Each was a **valid encoding whose meaning we assumed**. Four
were extremes, which is the worst place for them in a severe-weather model,
because an extreme is precisely what the model is built to notice — a
sentinel becomes the strongest signal in the dataset.

The fifth is worse still: `NaN` filled with `0` does not just add noise, it
**inverts the physical meaning** of a channel that gates storm initiation.

## Five more, and the pattern widening

| # | where | the value | read as | why it was dangerous |
|---|---|---|---|---|
| 6 | MERIT Hydro download | HTTP `200` | "the file exists" | the site had moved and returned an identical 2172-byte relocation page for **every** URL, including a filename that was invented to test it |
| 7 | MERIT Hydro raster | `-9999` at 99.95% of cells | a real elevation | correctly flagged by the sentinel scan; resolved by honouring the declared nodata rather than the array's contents |
| 8 | `tests/test_service_threads.py` | a green test | "the service works" | the test called `pyresample.kd_tree.resample_nearest` while the service goes through **satpy's** resample path. Green suite, aborting production |
| 9 | `EvalConfig(geometry="point")` on basins | POD `0.80` | "the flood head is good" | basins have the same array rank as gauges, so it runs. Point geometry counts every element once — right for a gauge, wrong for a basin. **True area-weighted POD: 0.0385** |
| 10 | ESA WorldCover COG overviews | a valid land-cover class | "the majority class here" | the overviews are **nearest-subsampled, not mode-aggregated**, so a coarse-overview pixel is *one sample*, not a majority. Measured **87.4%** agreement with the true 4 km majority |

### Why 10 is worth the entry even though nothing was invented

Every code an overview returns is a real WorldCover class, so the map looks
correct at every zoom and no check on values would fire. The error is 12.6%
of cells and it is **biased, not random**: subsampling under-represents
*fragmented* classes — built-up and water — because those occur in small
patches that a sparse sample misses. Those are precisely the classes that
raise the curve number and mark where flash floods hurt people, so the bias
runs toward under-forecasting runoff in cities and along drainage lines.

The near-miss is worth recording too. The failure I expected was
`average`-built overviews, where averaging tree (10) and grass (30) yields
shrubland (20) — a real class, a plausible map, entirely fabricated. That is
*not* what WorldCover does (checked: every returned code is in the legend).
The actual defect was subtler and would have survived a legend check.

**Fix:** read at overview 1/4 (40 m) and compute the majority ourselves.
Measured agreement with the full-resolution 4 km majority: 1/2 and 1/4 both
**100.00%**, 1/8 99.31%, direct 1/64 87.4%. Cost is 81 MB/tile instead of
1296 MB, so the correct answer is also 16x cheaper than the naive-but-safe
one and 4x dearer than the wrong one.

### Why 9 is the worst of them

The first eight degrade or corrupt an input. Number 9 **improves the
reported score**, which means nothing looks wrong at any point:

* It ran without error or warning.
* The provenance line said `geometry: point (station locations)` — wrong, and
  printed in the result where it would be read as confirmation.
* The bias runs in the flattering direction, and specifically: it rewards a
  forecast that nails many tiny headwater basins and misses the one large
  valley. That is the exact failure mode a flood product must not have,
  scored as a success.
* 21x, on a headline metric.

There is no sentinel scan for this one. The defences that would have caught
it are the two now in place: **the geometry must be declared**, and a
declared basin geometry **must be given an explicit weighting** rather than
defaulting to one. A default here is what made instance 9 possible, so the
harness raises instead.

The same bug then had to be closed a second time in `bootstrap.py`, which
tested `cfg.is_point` and so let basin geometry through to compute FSS over
basin *index*. A rule enforced in one module is not enforced.

## Instances 10 and 11: the pattern is not about data

| # | where | the value | read as | why it was dangerous |
|---|---|---|---|---|
| 10 | ESA WorldCover COG overviews | overview level 1/64 | "the land-cover class here" | the overviews are nearest-SUBSAMPLED, not mode-aggregated, so a pixel is one sample rather than a majority: **87.4% agreement** with the true 4 km majority, biased against fragmented classes like built-up and water |
| 11 | a generated MOSDAC config | `"startTime": ""` | "unset, ignore this field" | to the search API it is an **unbounded range**. Matched **179,134 granules, 69.70 TB**. Syntactically valid JSON, semantically catastrophic |

### 10 is a near-miss worth more than the catch

The failure *expected* was average-built overviews fabricating shrubland (20)
from tree (10) and grass (30) — and a legend check finds that instantly,
because averaged codes mostly fall outside the legend. That check was run and
**passed**: every returned code was valid.

The actual defect passes a legend check and is invisible without comparing
against a full-resolution majority. The guard that would have caught the
failure imagined was not the guard that catches the failure present.

### 11 is the same shape with no data in it

Nothing decoded wrongly. A config file was written whose six dates lived in
a `_one_job_per_date` key the client never reads, on the assumption it would
iterate; it takes one `startTime`/`endTime` per file. The real fields were
left empty, and empty means *everything*.

The only thing between that and a 70 TB pull was the download client
happening to ask for confirmation. Nothing in this repo's tooling looked at
the config at all.

`nowcast_data/mosdac_config.py` now estimates before writing and **refuses**
above a granule limit, so the size is known at generation rather than at the
download prompt. It is validated against this exact incident: it reproduces
179,134 granules to within 3.2% and 69.70 TB to within 6.2%.

### The generalisation behind 11: KEEP is not TRANSFER

Auditing for other instances of the same category error found that **every
one of the three terms sizing the archive had it**, and each was discovered
separately:

| term | what it is | how it was used | ratio | found by |
|---|---|---|---|---|
| `boundingBox` = India | a keep-shaped filter | assumed to subset the download | **10.4x** | measuring 702 delivered files |
| `channels` = TIR1, WV | keep 2 of 6 | assumed to bound bytes fetched | **23.5x** | measuring dataset sizes in one file |
| `scans_per_active_day` = 24 | keep 24 of 48 | assumed to bound granules pulled | **2.0x** | auditing the configs |

```
planned:  8,000 scans x  43 MB =   336 GB
actual : 19,200 scans x 448 MB = 8,400 GB       25x
```

Not three mistakes — **one mistake made three times.** Each term describes
what we want to *end up with*, and each was silently used to size what
crosses the wire. They are only the same number when the source can subset
on that axis, and MOSDAC subsets on none of them: not area, not dataset, not
time of day.

The general form: **a filter expresses intent, not necessarily a reduction in
what is transferred.** Before a filter is allowed into a size estimate, it
has to be demonstrated that the *source* honours it — which is a measurement,
not a reading of the API docs. All three were caught by measurement and none
by reasoning.

## A THIRD failure mode: the under-diagnosed sentinel

> **A sentinel may occupy a RANGE, not a value. A fix that clears the flag is
> not the same as a fix that clears the defect.**

Instance 3 was recorded as "INSAT count->K LUT clamp, 180.09 K". It was fixed
by masking count 1023, the off-disk fill index. The flag cleared. The scan
went quiet. **The defect did not go away**, and the under-diagnosis survived
nine subsequent instances and a config freeze before anyone looked again.

The LUT does not saturate at one index. It saturates over a PLATEAU:

| channel | plateau | leaked through as a real measurement |
|---|---|---|
| TIR1 | counts 921–1023 | 35,489 cells (0.449% of the scan) |
| TIR2 | counts 922–1023 | 23,312 (0.295%) |
| WV | counts 996–1023 | 0 — *coincidence, not design* |
| MIR | counts 983–1023 | **89,929 (1.139%)** |

Every one of those cells decoded to the coldest value in the scene, which is
exactly what a convective-intensity model keys on. Three of the four frozen
channels were affected. WV showed zero only because that scan's plateau
happened to contain nothing but fill.

**Why it stayed hidden.** Masking the fill index removed the largest pile-up,
so the remaining 0.3–1.1% no longer stood out against it. The check that
found instance 3 kept passing, because it was asking "is there a suspicious
pile-up?" and the answer had become no — while the values it was there to
catch were still in the data.

**What makes this distinct from the adjacent check.** There the guard tested
something *beside* the thing it protected. Here the guard tested exactly the
right thing and the DIAGNOSIS was too narrow: a range was read as a point.
No amount of pointing the guard more carefully would have helped.

**The operational counter.** When a sentinel is found, establish its EXTENT
before fixing it, not just its value: how many encodings map to it, and is
the mapping saturated near the boundary? For a lookup table that means
reading the table. `lut_clamp_value` derives the plateau from each file's own
LUT, which is also why it needed no per-channel constant — the clamp value
differs (TIR1 179.86, TIR2 179.93, WV/MIR 179.69) and a constant would have
been wrong three times out of four.

**Checked and clear elsewhere.** SEVIR's VIL encoding is an analytic formula,
strictly increasing on bytes 6–254 (253 → 77.25, 254 → 79.26), so byte 255
really is a single-value sentinel and cannot be the last index of a saturated
tail. The pretrained checkpoint is not contaminated by this. VIL does have a
plateau at 0.0 across bytes 0–5, but that is the published equation's own
definition of "no VIL" rather than an artefact — a censoring worth knowing
when thresholding, not a decode bug.

## A SECOND failure mode: the adjacent check

Distinct from the sentinel pattern and worth its own name:

> **The check passes because it tests something adjacent to what it protects.**

Not a wrong threshold, not a bad value — a guard aimed slightly beside the
thing it guards, so it reports health about code that is not the code at
risk. Three instances:

| # | the guard | what it tested | what it protected | result |
|---|---|---|---|---|
| D | the re-pull config generator | the per-day COUNT shortfall | granules that were missing **or corrupt** | three corrupt files sat on count-complete days and were never re-pulled |
| A | `test_service_threads` | `pyresample.kd_tree.resample_nearest` | `ingest_scan`, which goes through **satpy's** resample path | green suite, aborting production |
| B | `_has_network()` in the ERA5 tests | `open_store()` — Zarr **metadata** | tests that read a data **chunk** | guard passed, suite hung 25 min with no signal |
| C | zenith banding in the alignment gate | offsets with band-masked **cloud and rain** | displacement, which needs the rain field whole | every interior band returned (0,0) — a gate that cannot fail |

C is the worst of the three and was caught only because it was tried against
a synthetic with a known answer. Masking both fields to a band means any
shift moves the rain off the mask, so zero offset always wins. It would have
turned the gate — the thing that clears all Indian training data — into
something that always passes.

**What distinguishes this from a sentinel bug:** there is no bad value
anywhere. Every function is correct in isolation. The defect lives in the
relationship between the check and the thing checked, so reviewing either
one alone finds nothing.

**The operational counter:** a guard must exercise the *production call
path*, not a call that resembles it — and where the guard protects a
decision, run it against an input whose answer is known independently. C was
found by a synthetic with a planted (-5, -5) displacement; A and B were
found in production and in a 25-minute hang.

## The rule, operationally

Before a new source is used for anything:

**1. Sentinel scan — mechanical.**
`nowcast_data.sentinels.detect_pileups` on every channel. It flags any single
value holding an anomalous share, scored on **isolation** (gap to the rest of
the distribution in robust units) rather than mass — because a rain field
legitimately piles 85% of its pixels at exactly 0.0. Wired into the INSAT and
IMERG checkers; wire it into every new one.

**2. Physics-convention check — per field, against documentation AND data.**
Reading the docs is not enough and neither is eyeballing the data; do both,
and record the answer. For every field establish:

- **units** — and verify the magnitude is physically plausible
- **sign convention** — is CIN positive or negative? is the flux
  eastward-positive? Verify against a known physical case, not intuition
- **coordinate ordering** — latitude north-first or south-first, level
  top-down or bottom-up, longitude 0–360 or −180–180
- **what missing means** — and specifically whether it is *missing* or a
  *physical statement*. CIN's NaN is a physical statement. Filling it as
  missing inverts it.
- **the fill value** — its numeric value, and what it decodes to

Record the verified answer in the module, next to the code that depends on
it. See the `VERIFIED CONVENTIONS` block in `nowcast_data/era5.py`.

**3. Range checks on robust percentiles, not extremes.**
Absolute min/max are contaminated by exactly the artefacts above. Bound p1 and
p99.9 instead, report the extremes as context, and flag saturation separately.
See `nowcast_data/insat.py`.

**4. For a PYRAMID or any downsampled product, ask how the levels were
built.** Overviews, coarsened grids and pre-aggregated products all answer
"what is here?" at a resolution you did not choose. Averaging categorical
data fabricates classes; subsampling it biases against fragmented ones.
Neither shows up as an invalid value. Compare one tile against a majority
computed from full resolution before trusting a pyramid level.

**6. For a new GEOMETRY, not just a new field.** Before scoring anything on
a new spatial unit, ask what one element of it *is*, and whether counting
elements equally is a physical statement or an accident of array shape. If
the elements differ in size, population or duration, equal counting is a
silent weighting decision — make it explicit and record it in the result.

## Why this earns its cost

Every one of the five was found by hand, after the data was already in use,
and each took a diagnostic session to trace. The scan is seconds. The
convention check is a few minutes per field, once per source, and it is
written down afterwards so nobody repeats it.

Five for five is not coincidence — it is the base rate. Budget for a sixth.
