"""Rank monsoon days by convective activity, to scope the INSAT archive.

Why this exists
---------------
Continuous monsoon coverage is ~15 TB and overwhelmingly non-convective: at a
3.8e-4 base rate most of it is the model watching nothing happen. SEVIR is
10,000 EVENTS, not years of continuous data, and it works.

So the archive is event-sampled: rank days by convective activity from IMERG
(small, no MOSDAC), then request INSAT only for high-activity days plus a
deliberate, documented sample of null days so the base rate stays honest and
the null test set survives.

Sampling: a few granules per day is enough to RANK days. Full half-hourly
resolution is only needed for the days that make the cut, to choose scan
times.
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nowcast_data.alignment_gate import scene_cellularity
from nowcast_data.imerg import fetch, read_precipitation

BOX = (62.3, 5.8, 101.7, 39.5)

ap = argparse.ArgumentParser()
ap.add_argument("--start", default="2024-06-01")
ap.add_argument("--end", default="2024-09-30")
ap.add_argument("--hours", default="03,09,15,21",
                help="UTC hours to sample per day (:30 granule of each)")
ap.add_argument("--dest", default="data/imerg_survey")
ap.add_argument("--out", default="data/imerg_survey/activity.csv")
a = ap.parse_args()

hours = [int(h) for h in a.hours.split(",")]
days = pd.date_range(a.start, a.end, freq="D")
print(f"surveying {len(days)} days x {len(hours)} granules = {len(days)*len(hours)} granules",
      flush=True)

rows = []
for i, d in enumerate(days):
    per_day = []
    for h in hours:
        t = f"{d:%Y-%m-%d}T{h:02d}:30"
        try:
            p = fetch(t, a.dest)
        except Exception as e:
            print(f"  {t} FAIL {type(e).__name__}", flush=True)
            continue
        s = read_precipitation(p)
        per_day.append(scene_cellularity(s.rate_mmhr, s.lats, s.lons, BOX))
    if not per_day:
        continue
    rows.append(dict(
        day=f"{d:%Y-%m-%d}",
        # max over the sampled hours: a day is "active" if it had a burst,
        # not if it drizzled steadily
        score=max(c["score"] for c in per_day),
        cores=max(c["cores"] for c in per_day),
        heavy_pct=max(c["heavy_pct"] for c in per_day),
        wet_pct=float(np.mean([c["wet_pct"] for c in per_day])),
        max_mm_hr=max(c["max_mm_hr"] for c in per_day),
        n_granules=len(per_day)))
    if i % 10 == 0:
        print(f"  {d:%Y-%m-%d}  score {rows[-1]['score']:7.1f}  "
              f"({i+1}/{len(days)})", flush=True)

df = pd.DataFrame(rows)
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
df.to_csv(a.out, index=False)
print(f"\nwrote {a.out}  ({len(df)} days)")
print(df["score"].describe().to_string())
