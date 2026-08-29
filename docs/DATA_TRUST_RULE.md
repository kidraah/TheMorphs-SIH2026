# The standing rule for new data sources

> **Assume every new data source contains at least one value that decodes to
> something plausible and wrong. Do not trust the source until the sentinel
> scan and a physics-convention check have both been run against it.**

This is not a precaution against a hypothetical. It is the generalisation of
**nine** separate instances found on this project, by nine different
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

## Four more, and the pattern widening

| # | where | the value | read as | why it was dangerous |
|---|---|---|---|---|
| 6 | MERIT Hydro download | HTTP `200` | "the file exists" | the site had moved and returned an identical 2172-byte relocation page for **every** URL, including a filename that was invented to test it |
| 7 | MERIT Hydro raster | `-9999` at 99.95% of cells | a real elevation | correctly flagged by the sentinel scan; resolved by honouring the declared nodata rather than the array's contents |
| 8 | `tests/test_service_threads.py` | a green test | "the service works" | the test called `pyresample.kd_tree.resample_nearest` while the service goes through **satpy's** resample path. Green suite, aborting production |
| 9 | `EvalConfig(geometry="point")` on basins | POD `0.80` | "the flood head is good" | basins have the same array rank as gauges, so it runs. Point geometry counts every element once — right for a gauge, wrong for a basin. **True area-weighted POD: 0.0385** |

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

**4. For a new GEOMETRY, not just a new field.** Before scoring anything on
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
