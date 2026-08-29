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
