"""THE ALIGNMENT GATE, tiled and clustered on the scene.

Slope from pooled tiles (identified by within-scene tan(zenith) variation,
sd 0.247); constant from per-scene values with a between-scene standard error
(sd of scene-mean tan(zenith) is only 0.041, and the scene-level term u is
common to every tile of a scene). Tiles buy the slope; only scenes buy the
constant.
"""
from nowcast_data._threads import ensure_pinned_or_reexec  # noqa: E402
ensure_pinned_or_reexec()
import nowcast_data  # noqa: F401,E402

import argparse, glob, json, os, re  # noqa: E402
from datetime import datetime  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

SCAN_ROOTS = ["/Volumes/Untitled/alig"]
IMERG_DIRS = ["/Users/evad/nowcast/data/imerg_survey", "/Users/evad/nowcast/data/imerg"]
FNAME = re.compile(r"3RIMG_(\d{2}[A-Z]{3}\d{4})_(\d{4})_L1B")


def scan_time(p):
    m = FNAME.search(os.path.basename(p))
    return datetime.strptime(m.group(1) + m.group(2), "%d%b%Y%H%M") if m else None


def find_imerg(t):
    slot = t.replace(minute=0 if t.minute < 30 else 30, second=0)
    for d in IMERG_DIRS:
        hits = sorted(glob.glob(os.path.join(
            d, "**", f"*{slot:%Y%m%d}-S{slot:%H%M}00*"), recursive=True))
        if hits:
            return hits[0]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scans", nargs="*", default=[])
    ap.add_argument("--tile-px", type=int, default=128)
    ap.add_argument("--out", default="runs/tiled_gate.json")
    a = ap.parse_args()

    from nowcast_data.alignment_gate import measure_scene_tiled, run_gate_tiled
    from nowcast_data.grids import india_area
    from nowcast_data.imerg import ingest as imerg_ingest
    from nowcast_data.insat import ingest_scan, read_metadata, sub_satellite_longitude

    if a.scans:
        paths = a.scans
    else:
        paths = []
        for r in SCAN_ROOTS:
            paths += [p for p in glob.glob(os.path.join(r, "**", "3RIMG*.h5"),
                                           recursive=True)
                      if not os.path.basename(p).startswith("._")]
        paths = sorted(paths)

    area = india_area()
    lons, lats = area.get_lonlats()

    scene_tiles, skipped = {}, []
    for p in paths:
        t = scan_time(p)
        g = find_imerg(t) if t else None
        if not g:
            skipped.append((os.path.basename(p), "no matching IMERG"))
            continue
        try:
            arrays, chk = ingest_scan(p, ("TIR1",), strict=False)
            sub_lon = sub_satellite_longitude(read_metadata(p))
            rain, _ = imerg_ingest(g, strict=False)
            name = f"{t:%Y-%m-%d %H%M}Z"
            tiles = measure_scene_tiled(arrays["TIR1"], rain, lats, lons,
                                        scene=name, tile_px=a.tile_px,
                                        sub_lon=sub_lon)
            ident = [x for x in tiles if x.identifiable]
            scene_tiles[name] = tiles
            tz = [x.mean_tan_zenith for x in tiles] or [np.nan]
            print(f"  {name}  sub_lon {sub_lon:.2f}E  tiles {len(tiles):>3}  "
                  f"identifiable {len(ident):>3}  "
                  f"tan(zen) {np.nanmin(tz):.2f}..{np.nanmax(tz):.2f}")
        except Exception as e:
            skipped.append((os.path.basename(p), f"{type(e).__name__}: {e}"))
            print(f"  {os.path.basename(p)}  SKIPPED {type(e).__name__}: {str(e)[:70]}")

    res = run_gate_tiled(scene_tiles)
    print()
    print("=" * 72)
    print(f"TILED ALIGNMENT GATE: {res.verdict}")
    print("=" * 72)
    print(f"  scenes used     : {res.n_scenes_used}")
    print(f"  tiles used      : {res.n_tiles_used}")
    cy, cx = res.constant_px
    if np.isfinite(cy):
        print(f"  constant        : dy {cy:+.2f}  dx {cx:+.2f} px "
              f"= ({cy*4:+.1f}, {cx*4:+.1f}) km")
        print(f"  constant SE     : {res.constant_se_px:.3f} px "
              f"(95%: +/-{1.96*res.constant_se_px:.2f})")
        print(f"  sd(u) scene-level: {res.scene_level_sd_px:.3f} px")
    print(f"  parallax slope  : {res.parallax_slope_km:+.1f} km per unit "
          f"tan(zenith)   [12 km cloud top predicts ~12]")
    print()
    for r in res.reasons:
        print(f"  - {r}")
    if skipped:
        print(f"\n  {len(skipped)} skipped")
        for n, w in skipped[:5]:
            print(f"    {n}: {w}")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({
        "verdict": res.verdict, "constant_px": res.constant_px,
        "constant_se_px": res.constant_se_px,
        "scene_level_sd_px": res.scene_level_sd_px,
        "parallax_slope_km": res.parallax_slope_km,
        "n_scenes": res.n_scenes_used, "n_tiles": res.n_tiles_used,
        "reasons": res.reasons, "skipped": skipped,
        "scenes": {k: [vars(t) for t in v] for k, v in scene_tiles.items()},
    }, indent=1, default=float))
    print(f"\nwrote {a.out}")
    return 0 if res.verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
