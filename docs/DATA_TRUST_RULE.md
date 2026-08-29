# The standing rule for new data sources

> **Assume every new data source contains at least one value that decodes to
> something plausible and wrong. Do not trust the source until the sentinel
> scan and a physics-convention check have both been run against it.**

This is not a precaution against a hypothetical. It is the generalisation of
five separate instances found on this project, in five different sources, by
five different accidents. None of them raised. All of them decoded to
something a reasonable person would accept.

## The five

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

## Why this earns its cost

Every one of the five was found by hand, after the data was already in use,
and each took a diagnostic session to trace. The scan is seconds. The
convention check is a few minutes per field, once per source, and it is
written down afterwards so nobody repeats it.

Five for five is not coincidence — it is the base rate. Budget for a sixth.
