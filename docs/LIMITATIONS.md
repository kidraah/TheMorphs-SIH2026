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

## 4. End-to-end latency: measured on MPS, unmeasured at the real config

The differentiation claim against NWP is latency. The full chain — decode →
regrid → assemble → forward → postprocess — is now measured by
`scripts/probe_latency.py`, through the deployed service shape rather than
an in-process path the service cannot legally use.

Measured (dim 128, MPS, full India grid): **6.70 s end-to-end**, of which
decode+regrid was **40%**. So `model.forward()` in isolation was never the
answer, which is why the probe exists.

**What is still open:** the number at the real config. The two terms scale
differently —

| term | scales with |
|---|---|
| decode + regrid | nothing (fixed work per scan) |
| forward | dim, depth, token count |

— so the 40/60 split at dim 128 does **not** carry over to dim 384 / depth
10. The probe prints the split and names which side dominates; it must be
re-run on the rented CUDA box before any latency figure is quoted. At dim
128 / grid 192 ingest was 98% of the chain, which shows how far the ratio
moves with config alone.

**Resolved along the way:** the OpenMP conflict that made the service abort
is gone rather than ordered around — ingest runs in an exec'd child that has
never imported torch, so the model may be built on the GPU at boot like any
normal service. See [SERVICE_ARCHITECTURE.md](SERVICE_ARCHITECTURE.md).

## 4b. "INSAT WV is 8 km" is SATELLITE-SPECIFIC — and it is load-bearing

**Read this before extending anything to INSAT-3DS.**

A premise underpinning the whole SEVIR pretraining design is that INSAT
water vapour is 8 km while TIR is 4 km, so SEVIR's 2 km WV is degraded
2 km → 8 km → back onto the 4 km analysis grid. That premise is **true for
INSAT-3D and INSAT-3DR and FALSE for INSAT-3DS.**

| satellite | `IMG_WV` shape | WV native |
|---|---|---|
| INSAT-3D | 1408 × 1402 | 8 km |
| INSAT-3DR | 1408 × 1402 | 8 km |
| **INSAT-3DS** | **2816 × 2805** | **4 km** |

3DS upgraded the water-vapour channel to full resolution. Consequences:

- **A model pretrained on 8 km WV and deployed on 3DS throws away real
  resolution** in the one channel this project calls its cornerstone
  predictor (IWV variation).
- A model trained on 3DS WV and run on 3DR would expect 4 km moisture
  structure that physically is not there.
- It also broke satpy: no `Longitude_WV`/`Latitude_WV` arrays exist on 3DS
  because WV shares the main geolocation. See `docs/INSAT_FORMAT_CHECK.md`.

Current mitigation: **train on 3DR only.** 3DR spans 2016-10-11 to present
and covers every hindcast event, so nothing is lost today. Anyone extending
past 3DR's lifetime must revisit the degradation design first — this is not
a config tweak, it changes what the backbone learns.

## 4c. OPEN: cold-tail divergence between 3DR and 3DS is unexplained

On the one coincident pair measured (2025-08-01, 15 min apart, common India
footprint, 787,968 cells):

```
   p1     p5    p10    p25    p50    p75    p90    p95    p99  p99.9
+13.42  +7.66  +4.74  +2.51  +0.88  +0.62  +0.97  +0.91  +0.82  +0.91
```

The warm half is a clean +0.90 K offset. The cold tail diverges by 13.4 K.

The plausible benign explanation is convective evolution across the 15-minute
gap plus parallax from an 8° difference in sub-satellite longitude (74°E vs
82°E), since deep cloud tops are strongly view-angle sensitive.

**But that is a hypothesis, not a finding.** A genuine divergence in the
count→temperature LUT at the cold end is equally consistent with this data,
and the cold tail is precisely where the convective signal lives — the part
of the distribution the model is built to key on. The two satellites also use
different calibration methods (3D/3DR `LAB CALIBRATED`, 3DS `ONLINE
CALIBRATED`), which is a mechanism for exactly this.

Unresolved. Not blocking, because we are 3DR-only. It becomes blocking the
moment anyone extends past 3DR. To settle it: many coincident pairs,
preferably clear-sky scenes where cloud-top evolution cannot confound, plus a
direct comparison of the two LUTs at the cold end.

## 5. Sentinel values decoded as physical extremes — FIVE found, assume a sixth

**Promoted to a standing rule: see [DATA_TRUST_RULE.md](DATA_TRUST_RULE.md).**

Three times on this project a fill or clamp value has decoded to a
plausible-looking physical **extreme** rather than raising:

| source | value | decoded as | share |
|---|---|---|---|
| SEVIR VIL | byte 255 | 81.33 kg/m² — top of the entire range | 27% of a missing event |
| SEVIR / INSAT IR | int16 min | −327.68 °C — coldest possible cloud top | 7% |
| INSAT L1B | count→K LUT clamp | 180.09 K in **two** channels with different physics | 0.33% |
| INSAT-3DS | `Sun_Elevation` attr | `7.68e-76` → reads as "0 degrees" | scalar |
| ERA5 | CIN `NaN` | "no inhibition" if filled with 0 — **sign-inverted** | ~40% |

The fourth is the same pattern in metadata rather than pixels: denormal-small
rather than out of range, so `float()` accepts it. It gated a night scan
correctly *by luck* and would have let reflective checks run on a daytime
scene with unknown illumination.

This is the worst possible failure mode for a severe-weather model: an
extreme is exactly what the model is built to notice, so a sentinel becomes
the strongest signal in the dataset. None raised. Each was found by hand,
after the data was already in use.

`nowcast_data/sentinels.py` now makes the check mechanical, and it runs
inside both the INSAT and IMERG checkers. It flags any single value holding
an anomalous share, scored on **isolation** (gap to the rest of the
distribution, in robust scale units) rather than mass alone — because a rain
field legitimately piles 85% of its pixels at exactly 0.0.

**Standing rule: run it on every new data source before trusting that
source.** Assume IMERG and INSAT L1B each hide one we have not met yet.

## 6. The 0.80 FAR ceiling is a placeholder, not an analysis

`TrainConfig.far_ceiling = 0.80` is **arbitrary**. It was chosen because it
is materially better than the 0.997 an unconstrained SEDI optimum produces,
not because anyone has costed it.

What would justify a number: a false-alarm ratio ceiling follows from the
cost ratio between a missed event and a false alarm. If a missed cloudburst
costs `C_miss` and a false warning costs `C_fa`, the decision-theoretic
threshold is where the expected costs balance, which for a rare event with
base rate `p` gives an acceptable FAR of roughly

    FAR* ≈ 1 / (1 + (C_miss / C_fa) · (p / (1 - p)))

so the ceiling depends on both the cost ratio **and** the base rate — meaning
each head deserves its own, not one shared number.

The inputs we do not have:

- **Cost of a missed cloudburst.** Not purely monetary; the Kedarnath and
  Wayanad events carried casualties in the hundreds.
- **Cost of a false evacuation.** Direct cost, plus the harder-to-quantify
  erosion of response: a community that is evacuated repeatedly for nothing
  stops evacuating. That is a *dynamic* cost — it rises with each false
  alarm — which a single static ratio does not capture.
- **Who acts on the alert.** A district officer pre-positioning resources
  tolerates a far higher FAR than a public evacuation order. These are
  different products with different ceilings, from the same model.

Until those exist, **0.80 is a placeholder and should be described as one**.
At 45% false alarms — the constrained cloudburst point measured on SEVIR —
nearly one warning in two is wrong. That may well be acceptable for
resource pre-positioning and clearly is not for evacuation.

The right people to set this are IMD / NDMA, not us. Until they do, report
both operating points and let the reader see the trade rather than
presenting either as *the* answer.

## 7. CLOSED: cross-attention between the satellite and thermodynamic streams

The problem statement specifies cross-attention:

> *"A shared multi-modal spatiotemporal transformer network continuously
> analyzes real-time satellite grids ... against the IMDAA-derived
> thermodynamic baselines using **cross-attention mechanisms**."*

**Closed.** `CrossAttention` in `nowcast_model/backbone.py` takes queries
from the satellite stream and keys/values from a context stream held at its
own resolution; `DividedSpaceTimeBlock(..., cross=True)` interleaves it, and
weights come back shaped `(layers, B, satellite_tokens, context_tokens)`
with each row summing to 1 — a per-location statement of which thermodynamic
cell was consulted, and the model's actual computation rather than a
post-hoc attribution. `ModelConfig(context_channels=0)` keeps the old
behaviour, and supplying context to a model built without it raises.

**Still to wire:** the training dataset does not yet populate `context=`, so
the path is built but not exercised end-to-end in training. The service and
latency probe do supply it (`IngestWorker.era5`).

The original gap, kept for the reasoning:

**The pre-fusion model did not do this.** Every input entered through one
`nn.Conv3d(in_channels, dim, ...)` stem: satellite bands, reanalysis fields
and static channels are concatenated along the channel axis and mixed by a
single convolution. That is early fusion, not cross-attention.

This is a gap on two counts, and the second matters more than the first:

**1. It does not match the stated design.** A judge reading the problem
statement and then the architecture will find the mechanism missing.

**2. There is a real physical argument for it.** The streams have genuinely
different native resolutions and cadences:

| stream | native | cadence |
|---|---|---|
| INSAT TIR | 4 km | 30 min |
| INSAT WV | 8 km (3D/3DR) | 30 min |
| ERA5 thermodynamics | ~25 km | 1 h |
| DEM / hydrology | 90 m → static | never |

Early fusion forces all of them onto the 4 km analysis grid *before* the
model sees anything, which upsamples ERA5 by ~6x and spends 36 tokens
representing what is physically one value. Cross-attention would let the
fine, fast satellite stream **query** a coarse, slow thermodynamic stream at
its own native resolution — fewer tokens, no invented resolution, and the
attention weights become a readable statement about which thermodynamic
context mattered where, which feeds the XAI panel directly.

**Proposed shape** (not yet implemented): keep the satellite stream as the
query path at 4 km/patch 4, tokenise ERA5 at its own ~25 km grid (roughly
39x44 = 1,716 tokens over the India box, i.e. ~3% of the satellite token
count), and add a cross-attention sub-layer per block. Cost is small because
the key/value set is tiny relative to the queries.
