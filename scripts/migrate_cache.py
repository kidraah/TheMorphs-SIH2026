"""Re-write cache entries from a superseded fingerprint into the current one.

Needed because a fingerprint changed mid-ingest AFTER raw files had already
been deleted. Re-downloading would be the alternative; migration is possible
here because the stored arrays carry their own `finite` mask, so the values
survive independently of how NaN happened to be encoded.

What it can and cannot recover, stated rather than assumed:

    data, finite, channels, sub_lon, checks_passed   carried over exactly
    scan_time_utc, satellite                        derived from the filename
    reader                                          NOT recoverable -> "migrated"

`reader` is provenance for which decode path ran (satpy or the native LUT
fallback). Losing it on 176 granules is a real, small loss and it is recorded
as "migrated" rather than guessed, so a later audit can tell these apart from
freshly ingested entries.

    python scripts/migrate_cache.py --from 90e78e734db78158 --dry-run
"""
from nowcast_data._threads import ensure_pinned_or_reexec  # noqa: E402
ensure_pinned_or_reexec()
import nowcast_data  # noqa: F401,E402

import argparse, json, re  # noqa: E402
from datetime import datetime  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

STEM = re.compile(r"(3[DRS]IMG)_(\d{2}[A-Z]{3}\d{4})_(\d{4})_")
SAT = {"3RIMG": "INSAT-3DR", "3DIMG": "INSAT-3D", "3SIMG": "INSAT-3DS"}

# Warm-end LUT plateau, read from 12 surviving raw granules' own tables.
# Constant per (product, channel) for the IR channels; WV is deliberately
# absent, for two reasons:
#
#   * its warm clamp VARIES per file -- 8 distinct values across 3DIMG
#     (312.25..313.87) and 3 across 3RIMG (326.03..327.87) -- so no constant
#     is correct for it, which is exactly why the live decode reads each
#     file's own table instead;
#   * it does not matter. WV sees the upper troposphere: the maximum
#     observed across cached granules is 263.27 K, ~50 K below its own
#     plateau. Its warm plateau is also a single count. Masking it would be
#     a no-op and guessing at it would not.
#
# This table exists ONLY for migrating entries whose raw file is gone. Any
# granule whose raw survives is re-ingested instead, which reads the LUT.
WARM_CLAMP_K = {
    ("3RIMG", "TIR1"): 340.06, ("3RIMG", "TIR2"): 340.07, ("3RIMG", "MIR"): 339.79,
    ("3DIMG", "TIR1"): 340.08, ("3DIMG", "TIR2"): 340.02, ("3DIMG", "MIR"): 339.94,
}


def meta_from_stem(stem: str) -> dict:
    m = STEM.match(stem)
    if not m:
        return {"scan_time_utc": "", "satellite": "", "reader": "migrated"}
    t = datetime.strptime(m.group(2) + m.group(3), "%d%b%Y%H%M")
    return {"scan_time_utc": t.isoformat() + "Z",
            "satellite": SAT.get(m.group(1), m.group(1)),
            "reader": "migrated"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="src_fp", required=True)
    ap.add_argument("--cache", default=None)
    ap.add_argument("--ledger", default="runs/insat_ingest.json")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--delete-source", action="store_true",
                    help="remove the old fingerprint directory after migrating")
    a = ap.parse_args()

    from nowcast_data.paths import cache_dir
    from nowcast_train.insat_cache import InsatCacheConfig
    from nowcast_train.insat_ingest import _write_cache_entry

    cfg = InsatCacheConfig()
    root = Path(a.cache or cache_dir())
    src, dst = root / a.src_fp, root / cfg.fingerprint()
    if src == dst:
        print("source and target fingerprints are the same; nothing to do")
        return 0
    if not src.is_dir():
        print(f"[ERROR] no such fingerprint directory: {src}")
        return 2

    # Skip macOS AppleDouble sidecars. The external volume creates a "._x"
    # for every file, and an unfiltered glob reports 352 entries where there
    # are 176 -- which also made the failure counter read 176/176.
    files = sorted(f for f in src.glob("*.npz")
                   if not f.name.startswith("._"))
    print(f"migrating {len(files)} entries")
    print(f"  from {src.name}")
    print(f"  to   {dst.name}")
    if a.dry_run:
        print("  (dry run -- nothing written)")

    done, skipped, failed = 0, 0, []
    for p in files:
        try:
            with np.load(p, allow_pickle=False) as z:
                data, finite = z["data"], z["finite"]
                chans = [str(c) for c in z["channels"]]
                # The key was renamed when per_scan_fields was wired
                # through: "sub_lon" -> "sub_satellite_longitude". A
                # migration has to read every generation it might meet.
                sub_lon = float(z["sub_lon"] if "sub_lon" in z
                                else z["sub_satellite_longitude"])
                checks = bool(z["checks_passed"] if "checks_passed" in z
                              else z["physics_check_passed"])
                prior = {k: str(z[k]) for k in
                         ("scan_time_utc", "satellite", "reader")
                         if k in z}
            if list(chans) != list(cfg.channels):
                skipped += 1
                continue
            # Values where finite; NaN elsewhere. The mask is authoritative,
            # so the old fill value never has to be guessed at.
            arrays = {c: np.where(finite[i], data[i].astype(np.float64), np.nan)
                      for i, c in enumerate(chans)}
            # decode_version 3: the LUT saturates warm as well as cold.
            prod = p.stem[:5]
            for c in chans:
                warm = WARM_CLAMP_K.get((prod, c))
                if warm is not None:
                    # NOT `a` -- that is the argparse namespace in this
                    # scope, and shadowing it made the ledger update fail
                    # after the migration had already written every file.
                    band = arrays[c]
                    arrays[c] = np.where(band >= warm - 1e-6, np.nan, band)
            if not a.dry_run:
                _write_cache_entry(dst, cfg, arrays, sub_lon=sub_lon,
                                   checks_passed=checks,
                                   meta={**meta_from_stem(p.stem), **prior},
                                   stem=p.stem)
            done += 1
        except Exception as e:
            failed.append((p.name, f"{type(e).__name__}: {e}"))
    print(f"\n  migrated {done}, skipped {skipped} (channel mismatch), "
          f"failed {len(failed)}")
    for n, w in failed[:5]:
        print(f"    {n}: {w}")

    # The ledger keys on raw path; those raw files are gone, so mark the
    # migrated ones cached under the new fingerprint or ingest will try to
    # re-fetch files that no longer exist.
    lp = Path(a.ledger)
    if not a.dry_run and lp.exists() and not failed:
        led = json.loads(lp.read_text())
        stems = {p.stem for p in files}
        n = 0
        for k, v in led.items():
            if Path(k).stem in stems and v.get("state") == "cached":
                v["error"] = f"migrated from {a.src_fp}"
                n += 1
        lp.write_text(json.dumps(led, indent=1))
        print(f"  annotated {n} ledger entries as migrated")

    if a.delete_source and not a.dry_run and not failed and done == len(files):
        for p in files:
            p.unlink()
        src.rmdir()
        print(f"  removed {src}")
    elif a.delete_source:
        print("  NOT deleting the source: migration was not clean")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
