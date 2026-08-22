# SIH26077 — Complete Study Guide
## AI-Driven Hyper-Local Early Warning System for Severe Weather Nowcasting

This is your ground-up primer: the physics/math you need, what India currently uses to forecast weather, the AI/ML toolkit for nowcasting, and how to explain your solution's edge over the status quo. Written for a hackathon builder, not a PhD — deep enough to defend every claim in front of judges, no deeper.

---

## PART 0 — The PS in Plain Words

Strip the jargon and the ask is: **build one AI model that watches the sky in real time and tells you, 2–6 hours ahead, "this exact location is about to get a severe thunderstorm / cloudburst / flash flood."**

Three things make this hard, and each maps to a piece of the PS:

1. **Speed vs. accuracy trade-off.** Physics-based weather models (NWP) are accurate but slow — they solve fluid-dynamics equations on supercomputers and take hours. By the time the run finishes, the 2-hour warning window is gone. → *Solution: skip the physics simulation, learn the patterns directly from data (deep learning).*
2. **These events are small and fast.** A cloudburst might be 5 km wide and last 30 minutes. Coarse global/regional models (25–50 km grid) simply can't "see" something that small. → *Solution: fuse high-resolution satellite + reanalysis + terrain data on a fine grid.*
3. **Three different disasters, one shared cause.** Thunderstorms, cloudbursts, and flash floods aren't independent — a cloudburst *causes* a flash flood if the terrain channels the water badly. → *Solution: multi-task learning — one shared "brain" that understands atmosphere, with three specialized "output heads."*

Everything below builds toward being able to explain and defend that three-part logic.

---

## PART 1 — Foundational Science (only what you need)

### 1.1 The three ingredients of a storm — memorize this triangle
Every severe convective storm needs:

| Ingredient | Plain meaning | What the PS uses to measure it |
|---|---|---|
| **Moisture** (fuel) | Water vapor in the air column | Integrated Water Vapor (IWV) from satellite WV channel |
| **Instability** (energy) | Is the air "willing" to rise explosively? | CAPE (energy available) and CIN (the lid holding it down) |
| **Lift** (trigger) | What forces the air to start rising? | Low-level wind convergence, orographic lift, wind shear |

If you remember nothing else, remember: **no storm without all three simultaneously**. This triangle is literally the backbone of the PS's "Predictive Matrix."

### 1.2 Key physics/math concepts, explained simply

**Integrated Water Vapor (IWV)**
The total mass of water vapor in a vertical column of atmosphere above one square meter of ground, usually in kg/m² (numerically equal to mm of "precipitable water" if it all condensed). Rising IWV over a short time = moisture pooling = cloudburst risk. Math: it's just an integral of specific humidity over height —
IWV = ∫ q(z)·ρ(z) dz (surface to top of atmosphere). You don't need to compute this by hand; satellites/reanalysis give it to you as a derived field. You just need to know what a *sudden spatial+temporal spike* in it means physically.

**CAPE (Convective Available Potential Energy)**
A measure (J/kg) of how much energy a rising air parcel would gain if it kept rising, computed by comparing the parcel's temperature to the surrounding environment at every altitude (this comparison uses the **adiabatic lapse rate** — how fast a rising air parcel cools). High CAPE (>2000 J/kg) = explosive thunderstorm potential.

**CIN (Convective Inhibition)**
The energy needed to *push* a parcel up through a warm layer before it can rise freely (the "lid"). High CIN + high CAPE = energy is "loaded" but capped — when the lid finally breaks (via lift), the storm is often extreme. This CAPE/CIN interplay is exactly why the PS calls it out as "eroding CIN + high CAPE = prime indicator."

**Convergence**
When wind vectors from different directions meet at the surface, air has nowhere to go but up (mass conservation — this is the continuity equation in fluid dynamics, ∇·v = 0 for incompressible flow, so horizontal convergence forces vertical velocity). This is the mechanical "trigger" for lift.

**Vertical Wind Shear**
The change in wind speed and/or direction with height. Low shear → storms stay put and dump rain in one place (classic cloudburst setup). High shear → storms organize, move fast, and can rotate (severe thunderstorms, sometimes tornadic). This is why the PS uses shear to predict *storm motion*, not just occurrence.

**Cloud Top Temperature (CTT) & its drop rate**
Colder cloud tops = higher, more developed clouds (temperature decreases with altitude in the troposphere — the **environmental lapse rate**, ~6.5°C/km). A *rapid* CTT drop means a cloud is punching upward violently — a real-time signature of an explosive updraft, observable directly from satellite thermal-infrared without waiting for radar.

**Geopotential height & U/V wind components**
Geopotential height is essentially "the altitude of a given pressure level," used because atmospheric physics is easier to express in pressure coordinates than height coordinates. U (east-west) and V (north-south) wind components are the two numbers meteorologists actually store instead of "wind speed + direction" because they're easier to do math with (you can add/subtract/average them directly, then convert back).

### 1.3 Geography layer — why terrain matters for flash floods
- **Slope & elevation** determine how fast water accelerates downhill (steeper = faster, more destructive flow).
- **Drainage basins/watersheds** determine *where* water converges — the same rainfall over a bowl-shaped basin is far more dangerous than over flat, well-drained land.
- **Digital Elevation Model (DEM)**: essentially a grid of ground elevation values. Overlaying predicted rainfall onto a DEM lets you compute (even with simple hydrology, not full flood simulation) which valleys/streams will see the fastest rise — this is what turns "it will rain a lot here" into "this specific village will flood."
- India's terrain diversity matters a lot for your framing: Himalayan foothills (steep, orographic lift, flash-flood prone — Uttarakhand, Himachal), Western Ghats (extreme orographic rainfall), Gangetic plains (flat, different flood mechanism — river/drainage overload rather than terrain-channeled flash floods), and semi-arid interior peninsula (fewer but more damaging isolated cloudbursts).

### 1.4 The math/ML layer you'll actually touch
You don't need a PhD in these — you need working intuition:
- **Convolution (CNNs)**: sliding filters over grid data (like satellite images) to detect local spatial patterns (a moisture blob, a cold cloud-top patch).
- **Recurrence (LSTM/ConvLSTM)**: models that carry a "memory" forward through time steps — good for "how is this storm evolving frame to frame."
- **Attention / Transformers**: instead of only looking at nearby pixels (like CNNs), attention lets the model weigh *any* other position in the grid (or in time) when making a prediction — critical for capturing large-scale atmospheric patterns (e.g., a moisture plume 200 km away feeding into a local storm). "Cross-attention" (mentioned in your PS) specifically lets one data type (e.g., real-time satellite) attend to another (e.g., IMDAA baseline) to find relevant patterns between them.
- **Multi-task learning (MTL)**: one shared backbone (encoder) that learns general atmospheric representations, branching into multiple output heads (thunderstorm head, cloudburst head, flash-flood head). Why it's better than three separate models: (a) shared learning — signal that helps predict thunderstorms often helps predict cloudbursts too, so the model generalizes better with less data per task; (b) one inference pass instead of three = faster and cheaper, matching the "no computational bottlenecking" claim in your PS.

---

## PART 2 — What India Currently Uses (as of 2026)

Know this cold — it's your "current approach" baseline to compare against.

### 2.1 Institutional structure
- **IMD (India Meteorological Department)**, under the **Ministry of Earth Sciences (MoES)** — the national forecasting authority, issuing everything from seasonal outlooks to nowcasts.
- **NCMRWF (National Centre for Medium Range Weather Forecasting)** — runs India's medium-range NWP models.
- **IITM Pune (Indian Institute of Tropical Meteorology)** — research arm; e.g. it built and maintains IMD's current nowcast/warning web systems.

### 2.2 Observation infrastructure (the "eyes")
- **Doppler Weather Radar (DWR) network**: <cite index="2-1">IMD currently operates 39 DWRs across the country in X-, C-, and S-bands, with plans to expand the network to 70 systems by 2025–2026</cite>. A more recent figure from IMD leadership: <cite index="12-1">India went from 24 radars before 2014 to around 45 now, targeting 74 radars by end of 2026/early 2027 and eventually 126</cite>. Radars are the single most important current nowcasting sensor — <cite index="3-1">they detect rainfall, wind speed, and hailstorms up to 200 km, and this radar-based method is exactly how today's 1–3 hour nowcasting is done</cite>. **The gap for your PS**: 200 km range with ~40–70 radars leaves large parts of India — especially hilly/remote terrain in the Northeast, parts of central India, and the interior peninsula — with poor or no radar coverage. Satellite-first approaches (like your PS proposes) don't have this hard geographic limitation.
- **INSAT-3D/3DR satellites** (ISRO): provide Water Vapor and Thermal Infrared channel imagery used to track moisture and cloud-top cooling — <cite index="3-1">scientists use these to view cloud movement, sea surface temperature, and humidity from space, and to catch a system like a Western Disturbance as it approaches</cite>. Distributed via **MOSDAC** (ISRO's meteorological/oceanographic data archive).
- **Automatic Weather Stations (AWS)**: ground-truth point measurements, being expanded — <cite index="21-1">IMD announced installing 50 new AWS each in Delhi, Mumbai, Chennai, and Pune</cite>.
- **IMDAA Reanalysis**: a high-resolution (~12 km) reconstruction of India's past atmosphere (temperature, humidity, wind, geopotential height profiles), jointly built by NCMRWF, IMD, and the UK Met Office. This is your "historical baseline" dataset for training — it's what lets a model learn "what does the atmosphere typically look like right before a cloudburst" instead of learning from scratch.

### 2.3 Current forecasting method, in practice
- IMD runs a **seamless forecasting chain**: <cite index="12-1">seasonal forecasts at the start of a season, monthly, weekly (every Thursday for four weeks out), daily (7 days), and nowcasting — a very short forecast issued every 3 hours, valid for the next 3 hours</cite>.
- **Today's nowcasting method is fundamentally radar/satellite *extrapolation*** — track where a storm cell is now and where it's moving, and linearly (or simply) project it forward. This is a decades-old technique (radar-extrapolation nowcasting dates to the 1960s–70s) and struggles badly with **convective initiation** — i.e., predicting a *brand-new* storm that hasn't formed yet, as opposed to just moving an existing one forward. This is precisely the gap deep learning is meant to close, and it's the single strongest "why AI, why now" argument you can make.
- IMD has begun folding in AI/ML at the observational-data-processing and nowcasting layer: <cite index="20-1">the IMD Director-General has stated AI is expected to improve extreme-weather prediction lead time by 3–6 hours, and IMD has invited research groups to study AI integration since ML is not yet prevalent in India's operational weather forecasting</cite>. So the PS is explicitly riding an active, acknowledged institutional gap — not a hypothetical one.
- A newer institutional layer: the **Multi-Hazard Early Warning Decision Support System (MHEW-DSS)** — <cite index="11-1">a Web-GIS platform integrating geospatial tech, meteorological observations, NWP output, and impact-based forecasting for cyclones, heavy rainfall, thunderstorms and other severe events</cite>, described elsewhere as an in-house, indigenously built system that <cite index="18-1">has saved roughly ₹250 crore and ended dependence on foreign vendors, cutting annual maintenance cost by about ₹5.5 crore</cite>. There's also **Mausam Gram**, <cite index="18-1">a citizen-facing hyperlocal forecasting platform in multiple Indian languages that has improved forecast accuracy by 15–20%, cut preparation time by three hours, and extended lead time from five to seven days</cite> — note this is about *daily* forecasts, not 2–6 hour severe-event nowcasting, which is exactly the niche the PS targets.
- In 2026, IMD also rolled out an <cite index="10-1">AI-powered monsoon forecasting platform (jointly with IITM and NCMRWF) predicting rainfall patterns up to four weeks ahead</cite>, and <cite index="11-1">indigenously-developed AI/ML applications from government institutes aimed at observational data processing, monitoring, and nowcasting of severe events</cite> — good evidence AI adoption is a live, funded national priority (relevant to "Mission Monsoon," a large funding program IMD leadership has referenced).

### 2.4 The honest gap (say this out loud in your pitch)
- **Radar coverage is geographically incomplete** — vast interior/hilly regions have no 200 km radar cover, so extrapolation nowcasting simply has no local data to extrapolate from.
- **Current nowcasting = extrapolation, not prediction of new storms.** It's good at "this storm will keep moving this way" and weak at "a storm is about to *form* here that doesn't exist yet."
- **Three hazards, three separate decision chains.** Thunderstorm alerts, cloudburst risk, and flood impact aren't natively fused into one hyper-local, terrain-aware probability map today — MHEW-DSS integrates *observations and NWP output* for decision support, but the PS is asking for an end-to-end, single-model, multi-hazard *predictive* engine, which is a more specific and more automated ask.
- **NWP latency.** Physics-based models remain accurate for larger-scale/longer-lead forecasts but are computationally heavy and update on the order of hours, not minutes — too slow for a 2–6 hour actionable warning that ideally updates continuously.

---

## PART 3 — The AI/ML State of the Art (what to actually build on)

This is an active global research area — you're not inventing an untested category, you're applying a well-studied one to an underserved geography. Know these families:

| Approach | How it works | Strength | Weakness |
|---|---|---|---|
| **Radar extrapolation (optical flow, TREC)** | Track motion vectors of existing radar echoes, project forward | Simple, fast, works well 0–30 min | Can't predict new storm formation ("convective initiation"); fails past ~1 hr |
| **ConvLSTM / PredRNN** | CNN (spatial) + LSTM (temporal) combined to learn spatiotemporal sequences directly from radar/satellite frames | First deep-learning approach to beat extrapolation; well-established | Tends to blur predictions at longer lead times |
| **U-Net-based (SmaAt-UNet, RA-UNet)** | Image-segmentation-style encoder-decoder, treats "predict next rain frame" like image-to-image translation | Efficient, good spatial detail | Can lose fine-grained detail at bottleneck; limited long-range context |
| **Transformer-based (Earthformer, Rainformer)** | Self-attention/cross-attention over space-time "cuboids" instead of only local convolutions | <cite index="23-1">Captures 3D spatial interactions and both local and global features better than pure CNN/LSTM approaches</cite> | Heavier compute, more data-hungry |
| **Generative models (DGMR, NowcastNet)** | GAN or diffusion-based generation instead of direct regression, to avoid "blurry average" predictions | <cite index="41-1">DGMR was shown by Google DeepMind to give skilful, high-resolution probabilistic nowcasts up to 2 hours</cite>; <cite index="37-1">NowcastNet unifies physical-evolution schemes with deep learning to give physically plausible nowcasts with sharp detail up to 3 hours, including extreme events</cite> | <cite index="26-1">DGMR specifically struggles with extreme-precipitation accuracy and can show unnatural motion/intensity at longer lead times</cite>; GANs are notoriously unstable to train |
| **Hybrid physics+AI (NowcastNet's evolution network, FuXi-Nowcast)** | Combine a physics-informed motion/evolution component with a learned generative refinement | <cite index="31-1">Shown to outperform pure NWP for extreme-precipitation nowcasting in real flood case studies</cite>; <cite index="40-1">FuXi-Nowcast integrates multi-source radar, surface, and 3D atmospheric fields in a multi-task Swin-Transformer, explicitly targeting the convective-initiation problem, and surpasses an operational 3-km NWP model on multiple metrics</cite> | Most complex to implement — good stretch goal, not MVP |

**Your PS's proposed architecture** — "shared spatiotemporal transformer backbone + multi-task output heads + cross-attention between real-time satellite and IMDAA baseline" — sits squarely in the current research frontier (closest analogues: Earthformer's cuboid attention + FuXi-Nowcast's multi-task Swin-Transformer approach). That's a strong, defensible technical position to cite to judges: you're not doing something exotic, you're doing something published-and-proven, applied to a real Indian data gap.

**A realistic MVP path for a hackathon timeframe:**
1. Start with a ConvLSTM or lightweight U-Net baseline on INSAT WV/TIR channels — get *something* predicting rainfall probability grids first.
2. Add the multi-task heads (thunderstorm / cloudburst / flash-flood) sharing that backbone.
3. Layer in DEM-based flood risk scoring as a post-processing step over your cloudburst probability map (simple slope/flow-accumulation logic is enough for a prototype — you don't need a full hydrological model).
4. If time allows, swap the backbone for a small transformer/attention block to demonstrate the "why AI beats extrapolation" story with an ablation (compare vs. simple optical-flow baseline — judges love a clear "before/after" chart).
5. Add an Explainable AI (XAI) layer (e.g., attention-map visualization or SHAP-style feature attribution) — the PS explicitly names this, and it directly answers "why should a disaster manager trust this."

---

## PART 4 — Positioning: why your solution beats "current approach"

Use this comparison directly in your pitch deck.

| Dimension | Current operational approach | Your proposed system |
|---|---|---|
| Method | Radar echo extrapolation + physics NWP | Data-driven spatiotemporal deep learning |
| New storm formation | Weak — assumes existing echoes | Learns precursor atmospheric *signatures* (IWV spikes, CAPE/CIN, CTT drop) before a storm visibly forms |
| Geographic coverage | Bound by ~40-70 radar sites, 200 km range each — gaps in hilly/remote India | Satellite-first (INSAT covers all of India uniformly) + reanalysis, so no radar "dead zones" |
| Hazard integration | Separate products/teams for thunderstorm, rainfall, flood impact | One multi-task model outputs all three from a shared understanding |
| Flood translation | Rainfall forecast handed off separately to hydrological/disaster teams | DEM fused directly into the model pipeline — output is location-specific flood risk, not just rainfall |
| Latency | NWP runs take hours; nowcast reissued every 3 hrs | Near-real-time inference (seconds once trained), continuously updatable |
| Explainability | Forecaster judgment + decision-support dashboards | Built-in XAI module showing *which* atmospheric signature triggered the alert |

Be honest in your pitch about what you're **not** replacing: you are not claiming to out-perform IMD's radar network at 0–30 minute lead times (extrapolation is genuinely excellent there), and you are not replacing full physics NWP for day-ahead forecasts. You're filling the **2–6 hour, hyper-local, radar-sparse-region gap** — that precision is what makes the pitch credible instead of overclaiming.

---

## PART 5 — Datasets & Tools to Get Hands-On With

- **IMDAA Reanalysis** — search "IMDAA reanalysis download NCMRWF" — historical baseline fields (temp, humidity, wind, geopotential).
- **MOSDAC (mosdac.gov.in)** — ISRO's portal for INSAT-3D/3DR data (WV, TIR channels).
- **IMD Mausam portal (mausam.imd.gov.in)** — live nowcast warnings, station data, satellite imagery, for reference/validation.
- **CartoDEM (ISRO Bhuvan) / SRTM (NASA, free, 30m global)** — DEM data for the flood-terrain layer.
- **ERA5 (ECMWF, free, global reanalysis)** — good practice dataset before you get IMDAA access, same underlying concepts.
- **SEVIR / Weather4Cast** — public international benchmark datasets used in almost every nowcasting paper above; great for prototyping your model architecture before swapping in Indian data.
- **Bhuvan (ISRO's geoportal)** — Indian geospatial/terrain data, useful for the DEM + drainage layer.

---

## PART 6 — Suggested Learning Roadmap

**Week 1 — Meteorology fundamentals (Part 1 above, deepened)**
- Learn CAPE/CIN, convergence, wind shear, IWV conceptually — don't do the calculus by hand, learn what each number *means physically* and how it drives your XAI story.

**Week 2 — India's current system (Part 2 above, deepened)**
- Read IMD/NCMRWF/MOSDAC "about" pages, understand DWR network, IMDAA, the nowcast issuance chain. This is what lets you say "here's exactly what we're improving on" instead of vague AI hype.

**Week 3 — ML architectures**
- Read (skim is fine) the DGMR paper (Nature, 2021), the NowcastNet paper (Nature, 2023), and one Transformer nowcasting paper (Earthformer). You don't need to reproduce them — you need to be able to explain, in one sentence each, why your architecture choice beats extrapolation and why multi-task beats three separate models.

**Week 4 — Build**
- MVP per the path in Part 3. Prioritize: get *a* working end-to-end pipeline (data → model → probability map → alert) over a perfect model. Judges reward completeness and a working demo over marginal accuracy gains.

### Video / course resources (topics to search for — pick current top results)
- "CAPE CIN explained meteorology" — short conceptual videos exist on YouTube from university atmospheric science channels.
- "ConvLSTM precipitation nowcasting explained"
- "Transformers explained" (any general deep learning explainer — 3Blue1Brown's neural network series is an excellent, non-domain-specific foundation for attention mechanisms)
- "IMD Doppler weather radar working" — several Indian news/explainer segments cover this well
- NPTEL (nptel.ac.in) — search "Atmospheric Science" or "Satellite Meteorology" for free IIT-taught foundational courses if you want structured video lectures with an Indian academic lens
- MOSDAC and IMD's own training/tutorial sections often have short explainer PDFs on how their satellite products are derived — good primary-source material for your report.

---

## Quick-reference cheat sheet (for your pitch / Q&A)

- **What's IWV and why does it matter?** Total water vapor in an air column; a rapid local spike = moisture pooling = cloudburst fuel.
- **What's CAPE/CIN?** CAPE = energy available for a storm; CIN = the "lid" holding it down. High CAPE + eroding CIN = imminent severe storm.
- **Why not just use radar?** Radar only covers ~40–70 sites at 200 km range each; huge parts of India, especially hilly/remote terrain, aren't covered. Satellite-based prediction is uniform across the whole country.
- **Why not just use NWP?** Too slow (hours-long runs) and too coarse (25-50km grids) to catch a 5km-wide, 30-minute cloudburst.
- **Why multi-task instead of three models?** Shared atmospheric understanding improves each individual prediction, and one inference pass is faster and cheaper than three.
- **Why is this better than current nowcasting?** Current nowcasting extrapolates *existing* storms; this predicts storms *before* they visibly form, using precursor atmospheric signatures.
- **What's the DEM for?** Converts "it will rain heavily here" into "this specific valley/village will flood," by modeling how terrain channels the water.
