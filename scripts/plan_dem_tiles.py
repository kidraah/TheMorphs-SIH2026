"""Which CartoDEM tiles to pull, in priority order.

Bhoonidhi caps CartoDEM at 20 tiles/day, so the order matters more than the
list: full AOI coverage is ~800 one-degree tiles (40 days), while the tiles
that actually carry the flash-flood signal are a small fraction of that.

Priority is by TERRAIN RELIEF over our hindcast events, because that is where
the flood head does work. Flat terrain contributes little: HAND and flow
accumulation over the Deccan or the Gangetic plain barely vary, so a DEM
there adds a near-constant channel.
"""
import json
from pathlib import Path

# Hindcast events (docs/DOWNLOAD_MANIFEST.md), with the terrain that matters.
REGIONS = [
    ("himalaya-uttarakhand", 29, 32, 77, 81, 1,
     "Uttarakhand 2021, Kedarnath-type. Steepest relief, orographic cloudburst."),
    ("himalaya-himachal", 30, 34, 75, 79, 1,
     "Himachal 2023, Amarnath 2022. High relief, narrow valleys."),
    ("western-ghats-kerala", 8, 13, 74, 78, 1,
     "Wayanad 2024, Kerala 2018. Orographic, landslide-coupled."),
    ("western-ghats-north", 13, 20, 72, 76, 2,
     "Mumbai 2017. Ghat escarpment plus urban catchment."),
    ("northeast-hills", 24, 28, 89, 96, 2,
     "Highest rainfall on earth (Cherrapunji). Steep, data-sparse."),
    ("deccan-urban", 12, 20, 74, 80, 3,
     "Hyderabad 2020, Bengaluru 2022. Low relief, urban drainage dominates."),
    ("chennai-coastal", 11, 14, 79, 81, 3,
     "Chennai 2015. Flat coastal, drainage-limited flooding."),
    ("indo-gangetic", 24, 30, 76, 88, 4,
     "Flat. Low DEM value, included last for completeness."),
]


def tiles(lat0, lat1, lon0, lon1):
    return [(la, lo) for la in range(lat0, lat1) for lo in range(lon0, lon1)]


def main():
    seen, out = set(), []
    for name, la0, la1, lo0, lo1, prio, why in sorted(REGIONS, key=lambda r: r[5]):
        for la, lo in tiles(la0, la1, lo0, lo1):
            if (la, lo) in seen:
                continue
            seen.add((la, lo))
            out.append(dict(lat=la, lon=lo, region=name, priority=prio,
                            tile=f"N{la:02d}E{lo:03d}", why=why))

    by_prio = {}
    for t in out:
        by_prio.setdefault(t["priority"], []).append(t)

    print(f"{'prio':>5} {'region':>24} {'tiles':>6} {'days @20/day':>13}  rationale")
    print("-" * 100)
    cum = 0
    for p in sorted(by_prio):
        regions = sorted({t["region"] for t in by_prio[p]})
        n = len(by_prio[p])
        cum += n
        why = next(t["why"] for t in by_prio[p])
        print(f"{p:>5} {', '.join(regions)[:24]:>24} {n:>6} {n/20:>12.1f}  {why[:44]}")
    print("-" * 100)
    print(f"{'':>5} {'TOTAL':>24} {len(out):>6} {len(out)/20:>12.1f}")
    print()
    print("CUMULATIVE download schedule at 20 tiles/day:")
    run = 0
    for p in sorted(by_prio):
        run += len(by_prio[p])
        print(f"  through priority {p}: {run:>4} tiles = {run/20:>5.1f} days")
    print()
    print("FIRST 20 TILES (day 1) -- steepest relief over the cloudburst events:")
    for t in out[:20]:
        print(f"  {t['tile']}  ({t['lat']}-{t['lat']+1}N, {t['lon']}-{t['lon']+1}E)  {t['region']}")

    Path("configs").mkdir(exist_ok=True)
    Path("configs/cartodem_tiles.json").write_text(json.dumps(
        {"note": "1x1 degree tiles, priority order. Confirm Bhoonidhi's tile "
                 "naming on the portal -- NxxEyyy is the common convention but "
                 "was not verified against the live catalogue.",
         "cap_per_day": 20, "tiles": out}, indent=2))
    print(f"\nwrote configs/cartodem_tiles.json ({len(out)} tiles)")


if __name__ == "__main__":
    main()
