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
                    help="fail any granule with a HIGH-suspicion pile-up")
    ap.add_argument("--halt-on-scan", action="store_true",
                    help="stop the whole run on the first such granule "
                         "(--strict-scan only fails that granule and continues)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1,
                    help="parallel processes; 8 is the tested setting")
    a = ap.parse_args()

    from nowcast_data.paths import cache_dir, report as paths_report
    from nowcast_train.insat_cache import InsatCacheConfig
    from nowcast_train.insat_ingest import (CacheRootConflict, check_cache_root,
                                            ingest_all, ingest_all_parallel)

    cfg = InsatCacheConfig()
    root = a.cache or str(cache_dir())
    from nowcast_data.paths import iter_data_files
    paths = sorted({str(p) for src in a.src
                    for p in iter_data_files(src, "*.h5")})
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

    runner = ingest_all_parallel if a.workers > 1 else ingest_all
    kw = {"workers": a.workers} if a.workers > 1 else {"halt_on_scan": a.halt_on_scan}
    led = runner(paths, cfg, root, a.ledger,
                     delete_raw=(not a.keep_raw and not a.dry_run),
                 strict_scan=a.strict_scan, write=not a.dry_run, **kw)
    print()
    print(led.report())
    seen = {}
    for k, v in led.results.items():
        for f in v.sentinel_flags:
            key = (f["channel"], round(f["value"], 2), f["suspicion"])
            seen.setdefault(key, []).append(f["fraction"])
    hi = {k: v for k, v in seen.items() if k[2] == "high"}
    n_hi_gran = sum(1 for v in led.results.values() if v.high_suspicion)
    print(f"\nSENTINEL SCAN")
    print(f"  {len(seen)} distinct (channel, value) pile-ups, of which "
          f"{len(hi)} are HIGH")
    print(f"  {n_hi_gran} granule(s) carry at least one HIGH "
          f"(a value can recur across granules, so these two counts differ)")
    # HIGH first. Sorting by fraction buried them: the flagged values sit at
    # 0.3-0.4% while the benign distribution modes are at 2-3%, so the top
    # ten by fraction showed nothing but [low].
    order = sorted(seen.items(), key=lambda kv: (kv[0][2] != "high", -max(kv[1])))
    for (ch, val, sus), fracs in order[:12]:
        mark = "  <-- HIGH" if sus == "high" else ""
        print(f"  {ch:>5} {val:>9.2f}  {100*max(fracs):>6.3f}% max  "
              f"in {len(fracs):>3} granule(s)  [{sus}]{mark}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
