# nowcast-eval

A scoring harness for severe-weather nowcasts — thunderstorm, cloudburst,
flash flood — at 30 min to 6 h lead times.

**It measures; it does not predict.** There is no model here, no fitting,
no randomness. Every number is a deterministic function of
`(predictions, observations, config)`. That is what makes it testable, and
it is why it gets built before the model.

## Quick start

```bash
uv venv --python 3.12
uv pip install -e ".[dev]"
.venv/bin/python -m pytest tests/ -q     # 29 known-answer tests
.venv/bin/python scripts/demo.py         # synthetic end-to-end scorecard
```

## Data contract

```
pred : (N, L, H, W) float, probabilities in [0, 1]
obs  : (N, L, H, W) truth — boolean, or continuous with EvalConfig.obs_threshold
mask : broadcastable bool, True = valid
```

`N` samples, `L` lead times, `H×W` grid. One head at a time — score the
thunderstorm, cloudburst and flash-flood heads separately, since their base
rates differ by orders of magnitude and pooling them is meaningless.

```python
from nowcast_eval import EvalConfig, evaluate

r = evaluate(pred, obs, EvalConfig(grid_km=4.0), lead_minutes=[30,60,120,240,360])
print(r.summary_table())
r.to_json("artifacts/run017.json")
```

## What it computes

| | |
|---|---|
| **Contingency** | POD, FAR, CSI, frequency bias, ETS, HSS — swept across thresholds, never a single cherry-picked one |
| **FSS** | Fractions Skill Score (Roberts & Lean 2008) at 0/25/50/100 km — partial credit for near-misses, which pixel CSI refuses to give |
| **Probabilistic** | Brier, Brier skill score vs climatology, reliability curve |
| **Baselines** | persistence, climatology, pySTEPS optical-flow extrapolation (optional) |
| **Multi-hazard** | all three MTL heads through one code path, with per-head regression detection |
| **Uncertainty** | bootstrap 95% CIs, day-blocked, plus a significance test for model comparison |
| **Rare events** | SEDI — base-rate independent, for heads where CSI degenerates |

Everything is broken out **per lead time**. A pooled number hides the only
thing that matters: where skill dies.

## Multi-hazard (the MTL heads)

```python
from nowcast_eval import evaluate_multi
r = evaluate_multi(preds, obss, cfgs, lead_minutes=LEAD)   # dicts keyed by hazard
print(r.summary_table())
print(r.compare(previous_epoch_result))                    # did a head regress?
```

The failure mode this exists to catch: a shared backbone reallocates
capacity toward whichever task dominates the gradient — which, with base
rates two orders of magnitude apart, is the thunderstorm head. Your
headline number can improve for six epochs while the cloudburst head, the
one the project exists for, gets steadily worse.

There is deliberately **no averaged "overall score"** — a mean of CSIs at
2e-2 and 2e-4 base rates is meaningless and would hide exactly that. Use
`worst_head()` when you need one number.

## Confidence intervals

```python
from nowcast_eval import attach_confidence_intervals, significantly_better
attach_confidence_intervals(result, pred, obs, cfg, n_boot=1000)
significantly_better(model_ci, persistence_ci)   # or is it inside the noise?
```

A cloudburst test set may hold a few hundred events. CSI on 152 events and
CSI on 150,000 print identically; these intervals make the difference
visible, and the harness warns when a sample is too thin to rank models at
all.

Two design points, both tested:

- **The resampling unit is the independent block**, never the pixel, the
  `(case, lead)` pair, or — when windows come from a shared storm day —
  the case. Pass `groups=day` and whole days are resampled together. Each
  level of correlation ignored makes the interval too tight, and the day
  level is the least visible: 400 windows from 12 storm days will report a
  confident interval that is wrong by a factor of several. Omitting
  `groups` asserts independence, and the harness says so in its warnings.
- Replicates are computed from per-case **sufficient statistics**, so 1000
  of them cost about one extra evaluation — and are exact, not approximate.

## Checkpoint selection

```python
score = result.worst_head_lower_bound("sedi")[1]   # not the mean, not CSI, not the point estimate
```

Three choices, each of which changes which epoch you keep:

- **Worst head, not the mean** — an average is gameable by trading the rare
  hazards for the common one.
- **SEDI, not CSI** — at identical forecast quality across heads, CSI
  spreads >8× and ranks the rarest head worst every time. A CSI minimum is
  a base-rate detector, not a quality detector.
- **Lower confidence bound, not the point estimate** — at a 2e-4 base rate
  the point estimate is noise, and selecting on it picks the epoch whose
  validation draw was luckiest.

A head too thin to measure returns `-inf` and **blocks** checkpointing
rather than being dropped from the minimum. Dropping it would report a
healthy score while blind to a head — and it is always the rarest, most
important hazard that goes blind first, because that is where the evidence
runs out.

## Data: `nowcast_data`

SEVIR loader, targeted at **INSAT-native resolution and cadence** rather
than SEVIR's own. Three decisions, each of which throws information away on
purpose:

- **Per-channel resolution matching, not a uniform resize.** INSAT TIR is
  4 km but WV is 8 km. SEVIR water vapour therefore goes 2 km → 8 km → back
  onto the 4 km grid, so it occupies the analysis grid carrying only 8 km of
  real information. Resizing it straight to 4 km would let the model learn
  moisture gradients INSAT physically cannot deliver — and IWV variation is
  this project's cornerstone predictor, so it is the worst channel to get
  wrong. Tested: WV must come out blocky at 2×2, TIR must not.
- **Cadence matched to the scan interval.** SEVIR is 5-minute frames; INSAT
  full disk is 30 min. Five of every six frames are discarded, because a
  backbone trained on 5-minute motion learns evolution at a timescale the
  operational feed never shows it.
- **Samples carry their storm day**, and `day_groups()` hands it straight to
  `bootstrap_ci(groups=...)`.

Two limits the loader enforces or flags rather than hiding:

- **SEVIR caps pretraining at ~3 h lead**, not 6. An event is 4 hours; spend
  1 h on context and 3 h of targets remain. A config asking for more raises
  with an explanation. The far half of the 2–6 h window has to come from
  IMDAA/INSAT fine-tuning.
- **The VIL decode is unverified and warns.** SEVIR stores VIL with a
  non-linear uint8 encoding, so the placeholder scale does not recover
  kg/m². Confirm it before defining any label threshold. Related: India has
  no NEXRAD — if the Indian truth source is IMERG (~11 km) or IMD's gridded
  gauge product (~25 km), **label** resolution, not input resolution, is the
  binding constraint on "hyper-local".

## The judgment calls

The formulas are exact. The numbers are dominated by the choices *around*
them, so all of those live in `EvalConfig` — explicit, defaulted, recorded
in every result — rather than buried in the code:

- **Threshold** for binarising probabilities. Swept, not picked.
- **Neighbourhood radii** for FSS.
- **Base rate.** Reported beside every score, always. At 1e-4, "no
  everywhere" scores 99.99% accuracy and FAR becomes hypersensitive to a
  handful of false alarms. CSI without a base rate next to it is a lie.
- **Masked cells** dropped by default, never silently counted as no-event.
- **Label definition.** `obs_threshold` binarises continuous truth. IMD's
  operational cloudburst figure is ~100 mm/hr over 20–30 km² — confirm this
  per hazard with a meteorologist *before* the main training run. This one
  line determines what the model learns.

## Validation

`tests/` (76 tests) proves the ruler is straight by known-answer testing,
not by eyeballing plausibility:

- perfect forecast → POD 1, FAR 0, CSI 1, Brier 0 **exactly**
- climatology → BSS **exactly** 0.0 (else every reported BSS is wrong)
- disjoint fields at unit neighbourhood → FSS **exactly** 0 (analytic)
- window spanning both blobs → FSS **exactly** 1 (analytic)
- FSS identical across a 40×40 and 200×200 domain — measures the forecast,
  not the grid
- uniform field → neighbourhood fraction 1.0 including at corners, catching
  the classic edge-bias bug
- both fields empty → FSS `nan`, so a model that never fires cannot score 1.0
- masked cells excluded from fractions and from the contingency table
- bootstrap pooling from sufficient statistics reproduces the full
  recomputation **exactly**, checked against `contingency()`, `fss()` and
  `brier_score()` at every neighbourhood
- pooled CIs report N cases, not N×L case-lead pairs; duplicating a case
  across more lead times must not narrow the interval
- a regressed MTL head is flagged even when the other two improve
- SEDI holds across three orders of magnitude of base rate where CSI
  collapses by >20x; SEDI is 0 for a random forecast and `nan` (not 1) for
  a perfect one, where its false-alarm-rate term is undefined
- day-blocked intervals ignore duplicated cases within a day, and the
  unblocked version demonstrably narrows — the failure mode, pinned
- every fixture is checked to be capable of failing: `test_fixture_is_honest`
  asserts the quality knob is monotone and produces real misses and false
  alarms, after an earlier version seeded from `hash()` (randomised per
  process) with an inverted quality parameter and passed regardless

## Training

See `scripts/validation_hook.py`. Short version: you train on a **loss**
(differentiable, per batch); the harness scores per **epoch** and drives
checkpoint selection and early stopping. Loss improving does not mean more
storms caught — at these base rates a model can improve loss while learning
to say "no" everywhere. The harness is what catches that.
