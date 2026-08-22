# SIH26077 — Complete Term Glossary & Study Material
### For a CS student with zero atmospheric-science background

Goal: after reading this once, every term in the PS document should read like a normal ML feature-engineering spec, not like a foreign language. Organized by *when you'd touch it in your pipeline* — input data → derived features → model → output → evaluation — because that's how you'll actually think about it as a builder.

---

## 1. The big-picture vocabulary (forecasting itself)

| Term | Plain meaning | Think of it as (CS analogy) |
|---|---|---|
| **NWP (Numerical Weather Prediction)** | Solving physics equations (fluid dynamics + thermodynamics) on a supercomputer to simulate the atmosphere forward in time | A hand-coded physics simulator/game engine — accurate but expensive to run, like ray-tracing vs. a trained image model |
| **Nowcasting** | Very short-range forecasting, roughly 0–6 hours out | The "next-frame prediction" problem, but for the atmosphere instead of video |
| **Synoptic scale** | Large weather systems, hundreds–thousands of km (e.g. a monsoon depression) | Low-resolution, slow-changing signal |
| **Mesoscale** | Medium systems, ~2–2000 km (thunderstorm clusters, sea breezes) | Where your PS actually lives — this is the scale of thunderstorms/cloudbursts |
| **Convective initiation** | The moment a brand-new storm starts forming, as opposed to an existing storm continuing | The hardest class in your prediction problem — a "new object appearing," not an existing object moving |
| **Reanalysis** | A dataset built by combining historical observations with a model to produce a physically consistent, gap-free "best estimate" of the past atmosphere | Think of it as a *cleaned, imputed, and labeled* historical dataset — exactly what you want for supervised training |
| **Ground truth / verification** | The actual observed outcome (did it rain, did it flood) used to check forecast accuracy | Your `y_true` |

---

## 2. Data sources (where your tensors come from)

| Term | Plain meaning | ML/data context |
|---|---|---|
| **IMDAA Reanalysis** | India-specific high-resolution (~12 km) historical reanalysis dataset (NCMRWF + IMD + UK Met Office) | Your **historical training set** — gives you temperature, humidity, wind, geopotential height profiles for many past events, including past cloudbursts/storms to learn from |
| **INSAT-3D/3DR** | ISRO's geostationary weather satellites | Your **live/real-time input sensor** |
| **MOSDAC** | ISRO's data portal that distributes INSAT products | The **API/download endpoint** for your satellite data |
| **WV Channel (Water Vapor)** | Satellite sensor band that detects atmospheric moisture at mid-upper levels | One of your input image "channels" (like the R/G/B channels of a photo, but this one is "moisture") |
| **TIR Channel (Thermal Infrared)** | Satellite sensor band that measures emitted heat → cloud-top temperature | Another input "channel" — this one is "temperature," used to derive CTT |
| **QPE (Quantitative Precipitation Estimation)** | Rainfall amount estimated from satellite (or radar), not physical rain gauges | Often your **label/target** for the precipitation-intensity part of the task, since ground rain gauges are sparse |
| **DEM (Digital Elevation Model)** | A grid of ground-elevation values | A **static input layer** (doesn't change with time, unlike the weather layers) — used for the flood part of the model |
| **CartoDEM / SRTM** | Specific DEM products — CartoDEM is ISRO's Indian DEM, SRTM is NASA's free global one | Two options for sourcing your elevation grid; SRTM is easier to get started with |
| **NetCDF (.nc files)** | The standard file format almost all atmospheric/satellite/reanalysis data comes in — stores multi-dimensional arrays (time × lat × lon × altitude) with metadata | This is what you'll actually be opening in Python. Not a term from the PS, but you *will* hit this the moment you download real data |

---

## 3. Thermodynamics — "is the air willing to storm?" (the Instability / Energy features)

| Term | Plain meaning | ML/data context |
|---|---|---|
| **Specific Humidity** | Mass of water vapor per unit mass of air, at a specific altitude | A **3D feature** (lat × lon × altitude) — the raw ingredient CAPE/CIN/IWV are computed from |
| **IWV (Integrated Water Vapor)** | Specific humidity summed ("integrated") over the entire vertical column → one number per location | A **2D feature map** (lat × lon), collapsing the altitude dimension. A fast local spike = moisture pooling |
| **Lapse rate** | How fast temperature drops as you go up in altitude | The comparison baseline used to compute CAPE (see below) |
| **Adiabatic process** | A rising air parcel that cools/warms *without* exchanging heat with its surroundings, purely from expanding/compressing | The physical assumption behind "how much would this parcel cool if it kept rising" — you don't need the calculus, just the concept: rising air cools on its own |
| **CAPE (Convective Available Potential Energy)** | How much energy a rising air parcel would gain if released — computed by comparing a rising parcel's temperature to the surrounding air at every altitude | Your **"fuel gauge"** feature — a single scalar per location/time, strongly correlated with severe-storm probability |
| **CIN (Convective Inhibition)** | The energy needed to push a parcel through a stable "capping" layer before it can rise freely | A **"lid" feature** — high CIN suppresses storms even with high CAPE; the model needs to learn the *interaction* between the two, not treat them independently |
| **Troposphere** | The lowest layer of the atmosphere (surface to ~10-15 km) where all weather happens | Just context: this is the vertical extent your altitude dimension actually needs to cover — you don't need data above it |

---

## 4. Kinematics — "what's pushing the air around?" (the Lift / Trigger features)

| Term | Plain meaning | ML/data context |
|---|---|---|
| **U/V Wind Components** | Wind broken into East-West (U) and North-South (V) numbers, instead of "speed + direction" | Two clean input channels — easier for a network to process than polar (speed, angle) form, because you can add/average/differentiate them directly |
| **Geopotential Height** | The altitude at which the atmosphere reaches a given pressure level (e.g. "the 500 hPa surface is at 5500 m here today") | Defines the **3D grid topology** of your pressure-level data; also indirectly encodes high/low pressure systems |
| **Low-level Convergence** | Surface winds flowing toward a common point, forcing air upward since it has nowhere else to go | Computed as the (negative) **spatial divergence** of the U/V wind field — this is your **spatial trigger** feature for where a new storm might start |
| **Vertical Wind Shear** | How wind speed/direction changes with altitude | Computed as the **vertical gradient** of the U/V fields — high shear → storms organize and move; low shear → storms stay put and dump rain in one place (classic cloudburst setup) |

---

## 5. Observational signatures — "what does the storm look like as it happens?"

| Term | Plain meaning | ML/data context |
|---|---|---|
| **CTT (Cloud Top Temperature)** | Temperature at the top of a cloud, from the TIR satellite channel | A per-pixel value derived straight from your TIR channel |
| **CTT Drop Rate** | How fast CTT is falling over consecutive satellite frames | A **time-derivative feature** (dT/dt) — colder, faster-dropping cloud tops = a violently growing updraft, a real-time "storm is exploding right now" signal |
| **Reflectivity (dBZ)** | Radar's measurement of how much energy bounces back off precipitation — a proxy for rain intensity | Not named in your PS but shows up everywhere in nowcasting literature; if you ever bring in radar data, this is the value you'll see |

---

## 6. Terrain / flood layer

| Term | Plain meaning | ML/data context |
|---|---|---|
| **Slope** | Steepness of terrain, derived from the DEM | Faster runoff on steep slopes = faster, more dangerous flash floods |
| **Drainage basin / watershed** | The area of land where all rainfall funnels toward the same stream/valley | Determines *where* rainfall from a wide area converges into one dangerous flow — this is what turns a rainfall map into a location-specific flood risk map |
| **Flow accumulation** | A GIS calculation of how much upstream area drains through each point on the DEM | The simplest way to approximate flood-channeling behavior for a prototype, without needing a full hydrological simulator |

---

## 7. Model architecture terms (from your PS's "Technical Methodology")

| Term | Plain meaning |
|---|---|
| **Spatiotemporal** | Varying across both space (lat/lon/altitude) and time — i.e., your data is a 4D+ tensor, not a flat table |
| **Multi-modal** | Combining different *types* of input data (satellite images + reanalysis grids + static DEM) into one model |
| **Data fusion & alignment** | Resampling/reprojecting all your different data sources onto one shared grid (same resolution, same coordinate system, same timestamps) before feeding them to the model — unglamorous but essential preprocessing |
| **Transformer / Attention** | A neural architecture where the model learns to weigh *any* other position in the input (not just nearby ones) when making each prediction |
| **Cross-attention** | Attention where one data source (e.g. live satellite) is used to "query" relevance from another data source (e.g. IMDAA baseline) — lets the model compare "what's happening now" against "what's normal/expected" |
| **Multi-Task Learning (MTL)** | One shared backbone network, with separate output "heads" for each of your three targets (thunderstorm, cloudburst, flash flood). Shared learning improves each task and is faster than three separate models |
| **XAI (Explainable AI)** | Techniques to show *why* a model made a prediction (e.g. attention-map visualization, feature attribution) | Directly answers "why should a disaster manager trust this alert" |

---

## 8. Evaluation metrics you'll need (not in the PS text, but you'll need these to prove your model works)

| Term | Plain meaning |
|---|---|
| **CSI (Critical Success Index)** | Of all times a storm was forecast OR observed, what fraction did the model get right? The standard nowcasting accuracy metric — punishes both false alarms and misses |
| **POD (Probability of Detection)** | Of all storms that actually happened, what fraction did the model catch? |
| **FAR / SUCR (False Alarm Ratio / Success Ratio)** | Of all times the model raised an alert, what fraction were false alarms? |

Use these instead of plain accuracy — severe weather is rare, so a model that just always predicts "no storm" would score >99% accuracy while being useless. CSI/POD/FAR are what every nowcasting paper actually reports, and citing them will make your evaluation section look credible to judges.

---

## 9. Recommended study path (in order)

1. **COMET MetEd (UCAR)** — free account, search "Convective Weather" and "Satellite Meteorology" modules. This is the actual training material meteorologists use — visual, not equation-heavy. Best single resource for making CAPE/CIN/lift *click*.
2. **"How to read a Skew-T Log-P diagram"** (search YouTube) — the standard chart meteorologists use to visualize a full vertical profile (temperature, dew point, and where CAPE/CIN "areas" physically sit on the chart). Once this clicks, the whole thermodynamics section stops feeling abstract.
3. **Python libraries to get hands-on with:**
   - **xarray** — for opening and slicing the multi-dimensional NetCDF arrays (time × lat × lon × altitude) that IMDAA/INSAT data comes in
   - **MetPy** — computes CAPE, CIN, shear, etc. directly from raw temperature/humidity/wind data, so you don't have to implement the physics formulas yourself
   - **Cartopy** — plots your data on real geographic maps, needed for the dashboard/visualization side
4. **Then loop back to the ML side** (covered in your earlier study guide): ConvLSTM → U-Net → Transformer-based nowcasting papers (Earthformer, NowcastNet, DGMR).

You genuinely do not need a textbook. Between COMET MetEd's visual modules, one Skew-T explainer video, and MetPy's docstrings (which explain the physics inline as you call each function), you'll have working intuition for every term above within a few focused sessions.
