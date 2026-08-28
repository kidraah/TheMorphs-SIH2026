# Known limitations

Stated plainly, because a judge will find them anyway and it is better to
have named them first. Each has a stated densification or mitigation path.

## 1. Product resolution is 12 km, set by label availability — not by the 4 km inputs

The inputs run on a 4 km grid because that is INSAT TIR-1's native
resolution. The **labels do not**: GPM IMERG is 0.1° (~11 km), so every
gridded head is supervised at ~12 km and the output risk map is
piecewise-constant in 3×3 blocks of the 4 km grid.

Calling the product "4 km" because the inputs are 4 km would be a claim the
labels cannot support.

**Why not finer:** India has no NEXRAD-equivalent national radar mosaic.
The alternative gridded products are INSAT-derived (IMSRA, HEM) and are
excluded for a separate reason — see [LABELS.md](LABELS.md).

**Densification path:** IMD AWS/ARG station observations. Stations are
point-scale and resolve genuine cloudburst intensity (>100 mm/hr over
20–30 km²) that a 120 km² IMERG cell averages away. The architecture
already has a dedicated point head for exactly this, scored with point
geometry rather than on the grid. Denser station coverage improves the
product without any change to the model.

## 2. SEVIR pretraining supports no scientific claim

Every SEVIR number in this repo is a **pipeline-and-warm-start result, not a
meteorological result**, and must not be quoted as a headline figure without
this caveat attached.

- **Wrong physics.** SEVIR is CONUS convection: synoptically forced, over
  largely flat terrain. Indian monsoon convection is substantially
  orographically forced. There is no DEM analogue in SEVIR at all, so the
  flash-flood pathway is entirely untrained by it.
- **Wrong horizon.** A SEVIR event is 4 hours, so pretraining reaches ~3 h
  lead. The 2–6 h claim rests on Indian data at the far end.
- **Thin evidence.** The test split holds 2,065 events but only **9
  independent storm episodes** (val: 16). Both are below the ~20 at which
  the bootstrap itself becomes reliable. Confidence intervals there are wide
  and are reported as such.
- **The cloudburst head is mechanism-only.** SEVIR has no gauge network, so
  that head trains on pseudo-stations sampled from the VIL grid. It learns
  the sampling and sparse-scoring path and a sensible initialisation — no
  real station relationship. Checkpoints are stamped to say so.

What SEVIR *does* buy: a debugged end-to-end pipeline and a warm start for
the shared backbone. The scientific claim rests entirely on Indian data.

## 3. Flash floods are scored with the wrong geometry (open)

The flash-flood pathway is catchment-scale by definition — slope, drainage,
routed precipitation — but is currently scored on the 4 km pixel grid like
the atmospheric heads. A forecast can be right about which valley floods and
score badly for misplacing pixels within it. Needs basin-polygon aggregation
from the DEM.

## 4. End-to-end latency is unmeasured (open)

The differentiation claim against NWP is latency, and it is not yet
benchmarked. When it is, it must cover the full chain — decode → regrid →
inference → publish — not `model.forward()` in isolation, which is the
term least likely to dominate.
