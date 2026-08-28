# Label sources

The single most consequential set of choices in the project. What the model
learns is defined here, not in the architecture.

## Do not train against INSAT-derived rainfall products

**IMSRA and HEM are excluded.** Both are ISRO retrievals *derived from INSAT
TIR* — the same imagery that feeds the model's input. Training INSAT → IMSRA
teaches the network to reproduce ISRO's retrieval algorithm, not to predict
atmospheric evolution. The apparent skill would be high and largely
circular: the model would be learning a function of its own inputs.

This failure is silent. Nothing in a loss curve or a scorecard reveals it —
CSI, SEDI and FSS would all look healthy, because the model really is
predicting the target well. It is the target that is wrong.

If someone proposes IMSRA later because it is higher-resolution or easier to
obtain, that is the argument to re-read.

## The three heads

Note that the heads carry **two different geometries**. The harness supports
this via `EvalConfig(geometry=...)`; see below.

| head | label source | resolution | geometry |
|---|---|---|---|
| rain rate | GPM IMERG | 0.1°, 30 min | grid |
| extreme rain | IMERG > 99.9th pct per cell per month | 0.1°, 30 min | grid |
| cloudburst | IMD AWS/ARG stations, > 100 mm/hr | point | **point** |

### Why "extreme rain" and not "cloudburst" on the grid

IMD defines a cloudburst as **100 mm/hr over roughly 20–30 km²**. An IMERG
cell at 0.1° is about **120 km²** — four to six times the area of the event.
A real cloudburst is spatially averaged with its surroundings and lands
*below* the threshold before the model ever sees it.

The definition and the label resolution are therefore incompatible. Calling
a gridded IMERG-derived head a "cloudburst" head would be a claim the data
cannot support. It is named **extreme rain**, defined by a per-cell,
per-month 99.9th percentile — a quantity IMERG can actually express.

The percentile is computed **per cell and per month** rather than globally
because Indian rainfall climatology varies enormously by location and by
monsoon phase; a single global threshold would put nearly every event in the
Western Ghats and almost none in the interior.

### The cloudburst head is point geometry

Genuine cloudburst detection needs station observations: IMD automatic
weather stations and automatic rain gauges, thresholded at > 100 mm/hr.
These are irregularly spaced points, not a grid.

Consequences, all enforced by the harness:

- Score it with `EvalConfig(geometry="point")` and `(N, L, S)` arrays.
- **FSS is omitted** for this head. A neighbourhood filter over station
  *array index* is not a spatial neighbourhood — it is alphabetical-order
  smoothing with a kilometre label attached. The harness previously accepted
  station data reshaped into a degenerate `1 × S` grid and reported a
  plausible-looking FSS; that path now raises.
- Station density is uneven and sparse. A "miss" may mean no gauge was
  present rather than no cloudburst occurred. Expect the event count to be
  small enough that bootstrap intervals dominate the reading — which is what
  `worst_head_lower_bound` is for.

## Pretraining labels (SEVIR)

SEVIR's VIL is NEXRAD-derived vertically integrated liquid at 1 km, decoded
to **kg/m²** by the verified piecewise rule in `nowcast_data.sevir.decode_vil`.

It is **not a rain rate**. Do not put a 100 mm/hr threshold on it; converting
VIL to rainfall intensity requires a separate relationship that is not part
of this pipeline.

India has no NEXRAD equivalent, so nothing at 1 km will exist operationally.
This is one more reason pretraining transfers *representations*, not
thresholds — every label definition has to be re-derived on Indian data.

## Open item

The IMERG percentile thresholds and the station QC procedure should be
reviewed by a meteorologist before the main training run. Not the code — the
definitions. This page is what to hand them.
