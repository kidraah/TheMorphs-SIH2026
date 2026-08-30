"""THE ALIGNMENT GATE. Nothing is ingested until this passes.

Measures the displacement between INSAT cold cloud top and IMERG surface
rain, and decomposes it into

    constant  -- georeferencing error, independent of viewing geometry
    parallax  -- h * tan(satellite zenith), an expected consequence of
                 looking at a 12 km cloud top from geostationary orbit

A systematic offset has NO downstream symptom: training proceeds, the loss
falls, and the model learns a stable wrong cloud-to-rain displacement while
every score looks healthy.

    python scripts/run_alignment_gate.py
"""
from nowcast_data._threads import ensure_pinned_or_reexec  # noqa: E402
ensure_pinned_or_reexec()
import nowcast_data  # noqa: F401,E402

import argparse, glob, json, os, re  # noqa: E402
from datetime import datetime, timedelta  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

SCAN_DIRS = [
    "/Users/evad/PycharmProjects/PythonProject2/data/insat/alignment_gate",
    "/Users/evad/test_ff/3RIMG_L1B_STD",
]
IMERG_DIRS = ["/Users/evad/nowcast/data/imerg", "/Users/evad/data/imerg",
              "/Users/evad/PycharmProjects/PythonProject1/data/imerg"]
FNAME = re.compile(r"3RIMG_(\d{2}[A-Z]{3}\d{4})_(\d{4})_L1B")


def scan_time(path):
    m = FNAME.search(os.path.basename(path))
    if not m:
        return None
    return datetime.strptime(m.group(1) + m.group(2), "%d%b%Y%H%M")


def find_imerg(t):
    """The IMERG half-hour containing this scan time."""
    slot = t.replace(minute=0 if t.minute < 30 else 30, second=0)
    mins = slot.hour * 60 + slot.minute
    pats = [f"*{slot:%Y%m%d}-S{slot:%H%M}00*{mins:04d}*.HDF5",
            f"*{slot:%Y%m%d}-S{slot:%H%M}00*"]
    for d in IMERG_DIRS:
        for p in pats:
            hits = sorted(glob.glob(os.path.join(d, "**", p), recursive=True))
            if hits:
                return hits[0]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-scenes", type=int, default=0)
    ap.add_argument("--out", default="runs/alignment_gate.json")
    a = ap.parse_args()

    from nowcast_data.alignment_gate import measure_scene, run_gate
    from nowcast_data.grids import india_area
    from nowcast_data.imerg import ingest as imerg_ingest
    from nowcast_data.insat import ingest_scan, read_metadata, sub_satellite_longitude

    scans = []
    for d in SCAN_DIRS:
        scans += glob.glob(os.path.join(d, "**", "3RIMG*_L1B_STD*.h5"), recursive=True)
    scans = sorted(set(scans))
    print(f"{len(scans)} INSAT scans found")

    area = india_area()
    lons, lats = area.get_lonlats()

    pairs = []
    for s in scans:
        t = scan_time(s)
        g = find_imerg(t) if t else None
        if g:
            pairs.append((s, g, t))
    print(f"{len(pairs)} have a matching IMERG granule on disk")
    if not pairs:
        print("\nNo IMERG granules matched. Searched:")
        for d in IMERG_DIRS:
            print(f"    {d}  (exists: {os.path.isdir(d)})")
        return 2
    if a.max_scenes:
        pairs = pairs[:a.max_scenes]

    scenes, skipped = [], []
    for path, gran, t in pairs:
        try:
            arrays, chk = ingest_scan(path, ("TIR1",), strict=False)
            meta = read_metadata(path)
            sub_lon = sub_satellite_longitude(meta)
            rain, _ = imerg_ingest(gran, strict=False)
            so = measure_scene(arrays["TIR1"], rain, lats, lons,
                               scene=f"{t:%H%M}Z", sub_lon=sub_lon)
            scenes.append(so)
            print(f"  {t:%H%M}Z  sub_lon {sub_lon:.2f}E  dy {so.dy:+d} dx {so.dx:+d}  "
                  f"peak/zero {so.peak/max(so.zero,1e-9):.3f}  "
                  f"cold {so.n_cold:,} wet {so.n_wet:,}  "
                  f"{'identifiable' if so.identifiable else 'NOT identifiable'}")
        except Exception as e:
            skipped.append((f"{t:%H%M}Z", f"{type(e).__name__}: {str(e)[:90]}"))
            print(f"  {t:%H%M}Z  SKIPPED: {type(e).__name__}: {str(e)[:90]}")

    if skipped:
        print(f"\n  {len(skipped)} scene(s) skipped:")
        for name, why in skipped:
            print(f"    {name}: {why}")

    res = run_gate(scenes)
    print()
    print(res.report())

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({
        "verdict": res.verdict,
        "constant_px": res.constant_px,
        "parallax_slope_km": res.parallax_slope_km,
        "reasons": res.reasons,
        "scenes": [vars(s) for s in res.scenes],
        "skipped": skipped}, indent=2, default=float))
    print(f"\nwrote {a.out}")
    return 0 if res.verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
