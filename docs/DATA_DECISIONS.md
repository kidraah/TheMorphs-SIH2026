# Data source decisions

Deliberate choices, recorded so they read as decisions rather than
oversights — and so the reasoning survives the people who made it.

## Thermodynamics: ERA5, not IMDAA — deliberate

IMDAA is the better product on paper: **12 km hourly** against ERA5's
**~25 km**, and built for India specifically. It is not being used, for one
reason that outweighs resolution.

**IMDAA ends around 2018–2020. Our workhorse satellite, INSAT-3DR, spans
2016-10-11 to the present.** Using IMDAA would force two thermodynamic
sources across the training set, with a discontinuity partway through.

That is the same failure we just avoided on the satellite side by choosing
3DR alone over mixing 3D/3DR/3DS: a model given a source discontinuity can
learn *which era a sample came from* rather than the physics — stable in
training, invisible in the loss, worthless at inference.

ERA5 gives, against that:

- continuous coverage from 1940 to ~5 days behind real time — one source
  across the entire archive and into operations
- CAPE, CIN and TCWV **precomputed**, rather than derived by us from
  multi-level profiles with our own thermodynamic assumptions
- ARCO-ERA5 as public Zarr on Google Cloud: no approval queue, no CDS
  request-and-wait, and lazily sliceable, so we read the India box rather
  than downloading globally and cropping
- no NCMRWF transfer, which was the other queue on the critical path

**Cost accepted:** ~25 km against 12 km on the thermodynamic channels. The
instability and lift terms are synoptic-to-mesoscale quantities where that
matters less than it would for the moisture and cloud fields — which come
from INSAT at 4 km, not from reanalysis.

### IMDAA is not abandoned — it is a scoped validation

Keep it as a bounded exercise rather than an unused dependency:

1. Take one week of IMDAA/ERA5 overlap over India.
2. Compare the derived thermodynamic channels — CAPE, CIN, shear,
   moisture convergence — on the common grid.
3. Report whether ERA5's coarser resolution costs anything **measurable**
   on those channels.

That yields a defensible "we evaluated both and here is the number" rather
than either an unexamined assumption or a dependency nobody uses. It needs
the NCMRWF account, which is approved, so it is schedulable whenever the
thermodynamic path is otherwise done.

## Labels: IMERG, not INSAT-derived retrievals

See `docs/LABELS.md`. Short version: IMSRA and HEM are derived from INSAT
TIR, the same imagery that feeds the model's input, so training INSAT →
IMSRA learns ISRO's retrieval algorithm rather than atmospheric physics.

## Satellite: 3DR only

See `docs/LIMITATIONS.md` §4b and §4c. 3DR spans the whole archive, avoids
mixing instruments with different calibration methods, and sidesteps the
3DS water-vapour resolution change.

## INSAT-3DR scan cadence: both :15 and :45 — 30 minutes

Confirmed. The 3DR file we hold is at 2345Z, and GPI files were observed at
:15, so the satellite scans on the half hour at both slots.

An earlier config listed only `:45` times after I removed `:15` as
"unverified, inferred from a single file". That was right to flag but
over-corrected: the cadence is genuinely 30 minutes, which **doubles the
available temporal resolution** for the archive — 48 scans/day rather than 24.

Consequences:
- Archive planning uses 48 scans/day where full temporal coverage is wanted.
- The model's context window can be built at true 30-minute spacing, matching
  the SEVIR cadence we already subsample to, so no retiming is needed between
  pretraining and fine-tuning.
- The alignment-gate config still uses `:45` times only, deliberately: those
  are the ones whose IMERG pairings were measured for cellularity. No reason
  to change a gate config that is already specified.

## Archive scoping: event-sampled, not continuous

Continuous monsoon coverage is ~15 TB and overwhelmingly non-convective. At a
3.8e-4 base rate, most of it is the model watching nothing happen — and SEVIR,
which works, is ~12,000 events drawn from 526 storm days, not years of
continuous record.

So the archive is event-sampled: rank days by IMERG convective activity
(cheap, no MOSDAC), request INSAT only for high-activity days, plus a
**deliberate, documented sample of null days**.

The null days are not optional. Training only on active days inflates the
base rate the model sees, so it over-forecasts in operation, and the
verification protocol's null test set has nothing to score against. They are
sampled and recorded rather than dropped.

## Licensing: MERIT Hydro is CC-BY-NC 4.0 — non-commercial

MERIT Hydro (and MERIT DEM beneath it) are released under **CC-BY-NC 4.0**.
The NC clause is on a **core input layer**, not an optional extra: HAND and
upstream area feed the flash-flood head directly.

Consequences, recorded now rather than discovered later:

- Fine for SIH, for research, and for publication with attribution.
- **Not fine for a commercial product or a paid service** without separate
  permission from the authors.
- It is viral in the practical sense that matters: a trained model whose
  input channels were derived from MERIT is awkward to relicense, even
  though the weights themselves are not obviously a derivative work. That
  ambiguity is exactly what makes it worth flagging early.

**The escape hatch, if this is ever commercialised:** HydroSHEDS is
CC-BY 4.0 (no NC clause) and already supplies flow direction, flow
accumulation and a conditioned DEM. Only **HAND** is MERIT-exclusive, and
HAND is derivable from a conditioned DEM plus a drainage network — both of
which HydroSHEDS provides. So the non-commercial dependency is one derived
layer, not the whole hydrology stack, and it is replaceable with effort.

CartoDEM (ISRO/Bhoonidhi) has its own terms and is the more defensible
choice for an Indian operational system regardless of licensing.

### Why `upa` is taken from MERIT even though HydroSHEDS has accumulation

Deliberate: MERIT's `upa` is derived from the same conditioned DEM as its
`hnd`, so the two hydrology channels share a lineage. Mixing HydroSHEDS
accumulation with MERIT HAND would put two different DEM conditionings into
adjacent channels, and any disagreement between them would look to the model
like signal.

Note `n30e090` and the northern half of `n30e060` are mostly Tibet/China;
only the southern strips intersect the AOI.


## MERIT `dir` and `elv` are required; HAND cannot substitute for either

An earlier call was to skip `dir` and `elv` and work from `hnd` and `upa`
alone. That was wrong on both counts, and the reasoning is recorded here
rather than quietly overwritten, because both mistakes are easy to repeat.

**`dir` (flow direction).** The track routes on MERIT's own D8 field. It is
not derivable from `hnd` or `upa` — both were *computed from* `dir`, so
neither carries the routing back. `upa` gives the size of each cell's
catchment but not its shape or its outlet, which is what sub-basin
delineation needs.

**`elv` (hydrologically adjusted elevation).** HAND is height above the
*nearest drainage*, not elevation above datum. It therefore carries **no
downstream gradient**: two cells on opposite ends of a long river reach can
both sit 2 m above their local channel while the channel itself drops 200 m
between them. Channel slope — which every time-of-concentration method needs
— is a difference of elevations along the channel, and HAND differences
along a channel are close to zero by construction.

Consequence if it had been skipped: `channel_slope` would be nan, and every
arrival time with it. The code returns nan rather than substituting a
default slope, precisely so that this failure is visible instead of
producing confident wrong warning times.

Both are downloaded (all eight tile groups), and the flow-direction
convention is verified against `upa` — see LIMITATIONS #8.
