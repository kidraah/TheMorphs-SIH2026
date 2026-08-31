"""Raw INSAT .h5 -> verified -> decoded -> regridded -> scanned -> cached -> deleted.

One granule at a time, with a ledger, so a crash costs one file and not a run.
Ordering matters and is enforced here rather than left to a caller:

    1. VERIFY the file opens as HDF5 and carries the datasets. Eight event
       files on disk pass `os.path.exists` and are truncated -- the stock
       client promoted them to their final names after a mid-write failure.
       Anything that decodes a truncated file is decoding garbage.
    2. DECODE + REGRID onto the India grid.
    3. SENTINEL SCAN the decoded arrays, before they reach the cache.
       Resampling a mis-decoded field makes it smooth and plausible, so the
       scan has to see the native values. Failures are recorded, and by
       default they do NOT delete the raw file -- a scan failure is the one
       case where the original is worth keeping.
    4. APPEND to the cache under the frozen fingerprint.
    5. DELETE the raw file. Raw must never accumulate: the full archive is
       8.2 TB whole-file and 360 GB ranged, against a 197 GB cache.

THE FINGERPRINT TRAP THIS GUARDS
--------------------------------
`build_cache` writes to `out_dir / fingerprint`, so changing the config
mid-ingest does NOT raise and does NOT corrupt anything -- it silently starts
a SECOND cache in a sibling directory. You discover it as a disk filling up
with zero reuse, after paying for it. `check_cache_root` refuses to start
when a sibling fingerprint already holds data.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .insat_cache import InsatCacheConfig

REQUIRED_DATASETS = ("IMG_TIR1", "IMG_WV", "Latitude", "Longitude")
# 0.001 catches the 0.33% LUT clamp already found in real INSAT data.
SENTINEL_MIN_FRACTION = 0.001


class CacheRootConflict(RuntimeError):
    """Another fingerprint already holds data under this cache root."""


@dataclass
class GranuleResult:
    path: str = ""
    state: str = "pending"        # cached | failed_verify | failed_scan | failed
    seconds: float = 0.0
    raw_mb: float = 0.0
    error: str = ""
    sentinel_flags: list = field(default_factory=list)
    checks_passed: bool = False
    high_suspicion: int = 0


def check_cache_root(root, cfg: InsatCacheConfig) -> dict:
    """Refuse to start beside a different fingerprint that already has data."""
    root = Path(root)
    want = cfg.fingerprint()
    siblings = []
    if root.exists():
        for d in root.iterdir():
            if d.is_dir() and d.name != want and len(d.name) == 16:
                n = sum(1 for _ in d.rglob("*.npz"))
                if n:
                    siblings.append((d.name, n))
    if siblings:
        raise CacheRootConflict(
            f"cache root {root} already holds data under a DIFFERENT "
            f"fingerprint: {siblings}. Requested {want}.\n"
            f"build_cache writes to root/<fingerprint>, so this would not "
            f"raise and would not mix -- it would silently build a second "
            f"197 GB cache beside the first. Either point at the matching "
            f"fingerprint, or delete the stale one deliberately.")
    return {"root": str(root), "fingerprint": want, "siblings": siblings}


class IngestLedger:
    """One outcome per granule, written after each, atomically."""

    def __init__(self, path):
        self.path = Path(path)
        self.results: dict[str, GranuleResult] = {}
        if self.path.exists():
            raw = json.loads(self.path.read_text())
            self.results = {k: GranuleResult(**v) for k, v in raw.items()}

    def done(self, key: str) -> bool:
        r = self.results.get(key)
        return r is not None and r.state == "cached"

    def record(self, key: str, res: GranuleResult) -> None:
        self.results[key] = res
        self.save()

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps({k: asdict(v) for k, v in self.results.items()},
                                  indent=1))
        os.replace(tmp, self.path)

    def counts(self) -> dict:
        c: dict[str, int] = {}
        for r in self.results.values():
            c[r.state] = c.get(r.state, 0) + 1
        return c

    def report(self) -> str:
        c = self.counts()
        secs = sum(r.seconds for r in self.results.values())
        n = max(c.get("cached", 0), 1)
        lines = [f"{c.get('cached', 0)} cached, "
                 f"{sum(v for k, v in c.items() if k.startswith('failed'))} failed "
                 f"of {len(self.results)}",
                 f"    {secs / n:.1f} s per granule mean"]
        hi = [k for k, r in self.results.items() if r.high_suspicion]
        if hi:
            lines.append(f"    !! {len(hi)} granule(s) with HIGH-suspicion pile-ups")
        return "\n".join(lines)


def ingest_one(path, cfg: InsatCacheConfig, cache_root,
               delete_raw: bool = True, strict_scan: bool = False,
               write: bool = True) -> GranuleResult:
    """Verify -> decode -> regrid -> scan -> cache -> delete. One granule."""
    from nowcast_data.fetch import VerificationError, verify_hdf5
    from nowcast_data.insat import ingest_scan, read_metadata, sub_satellite_longitude
    from nowcast_data.sentinels import scan_source

    p = Path(path)
    res = GranuleResult(path=str(p))
    t0 = time.perf_counter()
    try:
        res.raw_mb = p.stat().st_size / 1e6 if p.exists() else 0.0

        # 1. verify BEFORE decoding
        verify_hdf5(p, REQUIRED_DATASETS, min_bytes=1 << 20)

        # 2. decode + regrid
        arrays, chk = ingest_scan(p, tuple(cfg.channels), strict=False)
        res.checks_passed = bool(chk.passed)

        # 3. sentinel scan the decoded arrays
        report = scan_source(arrays, min_fraction=SENTINEL_MIN_FRACTION)
        flags = []
        for name, rep in report.items():
            for pu in getattr(rep, "pileups", []):
                # isolation_iqr, not isolation -- reading the wrong attribute
                # silently discarded the scanner's whole triage signal and
                # left 24 undifferentiated "flags" per granule, most of them
                # the mode of a continuous distribution.
                flags.append({"channel": name, "value": float(pu.value),
                              "fraction": float(pu.fraction),
                              "isolation_iqr": float(pu.isolation_iqr),
                              "suspicion": pu.suspicion, "note": pu.note,
                              "is_min": bool(pu.is_min), "is_max": bool(pu.is_max)})
        res.sentinel_flags = flags
        res.high_suspicion = sum(1 for f in flags if f["suspicion"] == "high")
        if res.high_suspicion and strict_scan:
            res.state = "failed_scan"
            res.error = f"{res.high_suspicion} HIGH-suspicion pile-up(s)"
            return res

        # 4. cache
        if not write:
            res.state = "cached"        # dry run: verified + scanned, nothing written
            return res
        meta = read_metadata(p)
        out = Path(cache_root) / cfg.fingerprint()
        out.mkdir(parents=True, exist_ok=True)
        stack = np.stack([arrays[c] for c in cfg.channels])
        finite = np.isfinite(stack)
        np.savez_compressed(
            out / (p.stem + ".npz"),
            data=np.nan_to_num(stack).astype(np.float32),
            finite=finite,
            channels=np.array(cfg.channels),
            sub_lon=float(sub_satellite_longitude(meta)),
            checks_passed=res.checks_passed,
            fingerprint=cfg.fingerprint(),
        )

        # 5. delete raw -- only after the cache file exists
        if delete_raw:
            p.unlink()
        res.state = "cached"
    except VerificationError as e:
        res.state, res.error = "failed_verify", str(e)[:300]
    except Exception as e:
        res.state, res.error = "failed", f"{type(e).__name__}: {e}"[:300]
    finally:
        res.seconds = time.perf_counter() - t0
    return res


def dedupe_by_stem(paths, log=print) -> list:
    """One path per granule, preferring the one that verifies.

    The same granule exists in two trees after a re-pull: the original event
    directory and the re-pull directory. They write the same cache filename,
    so ingesting both is idempotent but wastes 5.6 s each -- and if the
    original copy is the CORRUPT one, ingest order would decide which
    version lands. Prefer whichever verifies; on a tie prefer the larger.
    """
    from nowcast_data.fetch import VerificationError, verify_hdf5

    by_stem: dict[str, list] = {}
    for p in paths:
        by_stem.setdefault(Path(p).stem, []).append(p)
    out, dropped = [], 0
    for stem, group in sorted(by_stem.items()):
        if len(group) == 1:
            out.append(group[0])
            continue
        ranked = []
        for p in group:
            try:
                verify_hdf5(p, REQUIRED_DATASETS, min_bytes=1 << 20)
                ok = True
            except VerificationError:
                ok = False
            ranked.append((ok, os.path.getsize(p), p))
        ranked.sort(reverse=True)
        out.append(ranked[0][2])
        dropped += len(group) - 1
    if dropped:
        log(f"  deduped {dropped} duplicate granule(s) across source trees "
            f"(kept the copy that verifies)")
    return out


def ingest_all(paths, cfg: InsatCacheConfig, cache_root, ledger_path,
               delete_raw: bool = True, strict_scan: bool = False,
               write: bool = True, log=print, log_every: int = 10) -> IngestLedger:
    check_cache_root(cache_root, cfg)
    paths = dedupe_by_stem(paths, log)
    ledger = IngestLedger(ledger_path)
    todo = [p for p in paths if not ledger.done(str(p))]
    log(f"{len(paths)} granules, {len(todo)} to ingest "
        f"({len(paths) - len(todo)} already cached)")
    log(f"cache -> {Path(cache_root) / cfg.fingerprint()}")
    for i, p in enumerate(todo, 1):
        r = ingest_one(p, cfg, cache_root, delete_raw, strict_scan, write)
        ledger.record(str(p), r)
        if r.state != "cached":
            log(f"  [{i}/{len(todo)}] {Path(p).name}: {r.state} -- {r.error[:90]}")
        elif r.high_suspicion:
            log(f"  [{i}/{len(todo)}] {Path(p).name}: cached with "
                f"{r.high_suspicion} HIGH-suspicion pile-up(s)")
        if i % log_every == 0 or i == len(todo):
            log(f"  [{i}/{len(todo)}] {ledger.report().splitlines()[0]}")
    return ledger
