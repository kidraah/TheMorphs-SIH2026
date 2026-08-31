"""Raw INSAT .h5 -> cache, one command, resumable.

    python scripts/ingest_insat.py --src /path/to/insat --dry-run
    python scripts/ingest_insat.py --src /path/to/insat

Fingerprint is frozen at 86fae373df9155e7. --dry-run verifies and scans
without writing the cache or deleting anything.
"""
from nowcast_data._threads import ensure_pinned_or_reexec  # noqa: E402
ensure_pinned_or_reexec()
import nowcast_data  # noqa: F401,E402

import argparse, glob, os, sys  # noqa: E402
from pathlib import Path  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, nargs="+", help="one or more roots of raw .h5 files")
    ap.add_argument("--cache", default=None, help="cache root (default NOWCAST_CACHE_DIR)")
    ap.add_argument("--ledger", default="runs/insat_ingest.json")
    ap.add_argument("--dry-run", action="store_true",
                    help="verify + scan only; write nothing, delete nothing")
    ap.add_argument("--keep-raw", action="store_true")
    ap.add_argument("--strict-scan", action="store_true",
                    help="treat any sentinel pile-up as a failure")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    from nowcast_data.paths import cache_dir, report as paths_report
    from nowcast_train.insat_cache import InsatCacheConfig
    from nowcast_train.insat_ingest import CacheRootConflict, check_cache_root, ingest_all

    cfg = InsatCacheConfig()
    root = a.cache or str(cache_dir())
    paths = []
    for src in a.src:
        paths += [p for p in glob.glob(os.path.join(src, "**", "*.h5"),
                                       recursive=True)
                  if not os.path.basename(p).startswith("._")]
    paths = sorted(set(paths))
    if a.limit:
        paths = paths[:a.limit]

    print(f"fingerprint : {cfg.fingerprint()}")
    print(f"channels    : {cfg.channels}")
    print(f"granules    : {len(paths)}")
    print(paths_report())
    try:
        check_cache_root(root, cfg)
    except CacheRootConflict as e:
        print(f"\n[REFUSED] {e}")
        return 2
    if not paths:
        print("nothing to do")
        return 0

    led = ingest_all(paths, cfg, root, a.ledger,
                     delete_raw=(not a.keep_raw and not a.dry_run),
                     strict_scan=a.strict_scan, write=not a.dry_run)
    print()
    print(led.report())
    seen = {}
    for k, v in led.results.items():
        for f in v.sentinel_flags:
            key = (f["channel"], round(f["value"], 2), f["suspicion"])
            seen.setdefault(key, []).append(f["fraction"])
    hi = {k: v for k, v in seen.items() if k[2] == "high"}
    print(f"\nSENTINEL SCAN: {len(seen)} distinct pile-up values, "
          f"{len(hi)} at HIGH suspicion")
    for (ch, val, sus), fracs in sorted(seen.items(), key=lambda kv: -max(kv[1]))[:10]:
        mark = "  <-- HIGH" if sus == "high" else ""
        print(f"  {ch:>5} {val:>9.2f}  {100*max(fracs):>6.3f}% max  "
              f"in {len(fracs):>3} granule(s)  [{sus}]{mark}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
