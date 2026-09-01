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

from nowcast_data.grids import india_area
from nowcast_data.paths import count_data_files, iter_data_files

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
                n = count_data_files(d, "*.npz")
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


# NaN encoding for the stored array. The finite mask is authoritative, not
# this value -- it exists only so the float array has no NaNs to trip
# downstream arithmetic. Chosen below any physical brightness temperature so
# that a consumer ignoring the mask gets an obviously-wrong number rather
# than a plausible one.
NAN_FILL_K = -999.0


# A LUT step wider than this makes a single count's cells look like an
# isolated spike. Measured: the flagged MIR values sat where the step is
# 7.0-16.7 K, while the genuine warm mode at 298 K sits where it is 0.146 K.
QUANTISATION_STEP_K = 2.0


def _quantisation_values(path, channels) -> dict:
    """Values that are isolated only because the LUT is coarse there."""
    import h5py

    out = {}
    try:
        with h5py.File(path, "r") as fh:
            for ch in channels:
                key = f"IMG_{ch}_TEMP"
                if key not in fh:
                    continue
                t = np.asarray(fh[key][:], dtype=np.float64)
                step = np.abs(np.gradient(t))
                out[ch] = t[step > QUANTISATION_STEP_K]
    except Exception:
        pass
    return out


def _near(value, candidates, tol: float = 0.05) -> bool:
    if len(candidates) == 0:
        return False
    return bool(np.min(np.abs(np.asarray(candidates) - value)) <= tol)


def _write_cache_entry(out_dir, cfg: InsatCacheConfig, arrays: dict,
                       sub_lon: float, checks_passed: bool, meta: dict,
                       stem: str = "entry"):
    """Write one granule, honouring every fingerprinted field that describes
    the bytes. Extracted so the coverage tests can call it directly."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stack = np.stack([np.asarray(arrays[c], dtype=np.float64)
                      for c in cfg.channels])
    finite = np.isfinite(stack)

    if cfg.nan_policy == "FILL_ZERO":
        filled = np.where(finite, stack, 0.0)
    elif cfg.nan_policy == "FILL_MEAN":
        m = np.nanmean(stack, axis=(1, 2), keepdims=True)
        filled = np.where(finite, stack, m)
    else:                                   # MASK -- the default
        filled = np.where(finite, stack, NAN_FILL_K)

    payload = {
        "data": filled.astype(np.dtype(cfg.store_dtype)),
        "channels": np.array(cfg.channels),
        "fingerprint": cfg.fingerprint(),
        "nan_fill": NAN_FILL_K,
    }
    if cfg.store_finite_mask:
        payload["finite"] = finite
    for f in cfg.per_scan_fields:
        if f == "sub_satellite_longitude":
            payload[f] = float(sub_lon)
        elif f == "physics_check_passed":
            payload[f] = bool(checks_passed)
        else:
            payload[f] = str(meta.get(f, ""))
    path = out_dir / f"{stem}.npz"
    np.savez_compressed(path, **payload)
    return path


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
        arrays, chk = ingest_scan(
            p, tuple(cfg.channels), strict=False,
            resampler=cfg.resample,
            radius_of_influence=cfg.radius_of_influence_m,
            area=india_area(shape=cfg.grid_shape, resolution_m=cfg.target_km * 1000,
                            lat_0=cfg.proj_lat_0, lon_0=cfg.proj_lon_0))
        res.checks_passed = bool(chk.passed)

        # 3. sentinel scan the decoded arrays
        report = scan_source(arrays, min_fraction=SENTINEL_MIN_FRACTION)
        # Near the cold end an inverted LUT is STEEP: at 185-197 K the MIR
        # table moves 7-16.7 K per count, so every cell of a single count
        # lands on one value whose neighbours are 7-16 K away. The isolation
        # score reads that as "isolated spike at an extreme" and it is
        # quantisation, not a sentinel. All six --strict-scan failures were
        # this. A true clamp is a value shared by MANY counts; quantisation
        # is one count standing alone.
        quantised = _quantisation_values(p, cfg.channels)
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
        for f in flags:
            if f["suspicion"] == "high" and _near(f["value"], quantised.get(f["channel"], ())):
                f["suspicion"] = "quantisation"
                f["note"] = ("isolated because the LUT step is wide here, not "
                             "because the value is a sentinel")
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
        _write_cache_entry(
            Path(cache_root) / cfg.fingerprint(), cfg, arrays,
            sub_lon=float(sub_satellite_longitude(meta)),
            checks_passed=res.checks_passed,
            meta={"scan_time_utc": str(meta.get("Acquisition_Date_Time", "")),
                  "satellite": str(meta.get("Satellite_Name", "")),
                  "reader": str(meta.get("_reader", ""))},
            stem=p.stem)

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


def _worker(args):
    """Top-level so it is picklable for a process pool."""
    path, cfg, cache_root, delete_raw, strict_scan, write = args
    return str(path), ingest_one(path, cfg, cache_root, delete_raw,
                                 strict_scan, write)


def ingest_all_parallel(paths, cfg: InsatCacheConfig, cache_root, ledger_path,
                        workers: int = 8, delete_raw: bool = True,
                        strict_scan: bool = False, write: bool = True,
                        log=print, log_every: int = 50) -> IngestLedger:
    """Ingest across processes. Embarrassingly parallel: one granule in, one
    npz out, no shared state.

    PROCESSES, not threads: the decode is numpy and h5py under the GIL, and
    the resample is pykdtree. Threads would serialise on exactly the part
    that costs the 5.6 s.

    No torch anywhere in this path -- asserted by
    test_the_ingest_side_never_needs_torch -- so the OpenMP conflict that
    forced the exec'd boundary in nowcast_serve does not arise here, and a
    plain pool is safe.

    SAFE TO RUN WHILE THE DOWNLOAD IS STILL GOING. The client writes to
    "<name>.part" and renames on completion, so a "*.h5" glob only ever sees
    finished files; there is no window where a half-written file is visible
    under its final name. The ingest simply processes what exists when it
    starts, and a second pass picks up whatever landed since -- which the
    on-disk skip check now makes cheap.
    """
    import concurrent.futures as cf
    import multiprocessing as mp

    check_cache_root(cache_root, cfg)
    paths = dedupe_by_stem(paths, log)
    ledger = IngestLedger(ledger_path)
    out_dir = Path(cache_root) / cfg.fingerprint()
    on_disk = {p.stem for p in iter_data_files(out_dir, "*.npz")}
    todo = [p for p in paths
            if Path(p).stem not in on_disk and not ledger.done(str(p))]
    log(f"{len(paths)} granules, {len(todo)} to ingest "
        f"({len(paths) - len(todo)} already cached), {workers} workers")
    if not todo:
        return ledger

    jobs = [(p, cfg, cache_root, delete_raw, strict_scan, write) for p in todo]
    ctx = mp.get_context("spawn")
    done = 0
    t0 = time.perf_counter()
    with cf.ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
        for path, res in ex.map(_worker, jobs, chunksize=1):
            ledger.record(path, res)
            done += 1
            if res.state != "cached":
                log(f"  {Path(path).name}: {res.state} -- {res.error[:80]}")
            if done % log_every == 0 or done == len(todo):
                el = time.perf_counter() - t0
                log(f"  [{done}/{len(todo)}] {el/done:.2f} s/granule wall, "
                    f"eta {(len(todo)-done)*el/done/60:.0f} min")
    ledger.save()
    return ledger


def ingest_all(paths, cfg: InsatCacheConfig, cache_root, ledger_path,
               delete_raw: bool = True, strict_scan: bool = False,
               write: bool = True, halt_on_scan: bool = False,
               log=print, log_every: int = 10) -> IngestLedger:
    check_cache_root(cache_root, cfg)
    paths = dedupe_by_stem(paths, log)
    ledger = IngestLedger(ledger_path)
    out_dir = Path(cache_root) / cfg.fingerprint()
    # Skip on what is ON DISK, not on the ledger. The ledger keys on the RAW
    # path, and after a migration the raw copy that produced an entry has
    # been deleted while a duplicate survives at a different path in another
    # tree -- so ledger.done() said False for 781 already-cached granules and
    # reported "0 already cached". Harmless at 518; on the 19,200-granule
    # archive that is hours of redundant decode and a much larger window for
    # a mid-run failure.
    on_disk = {p.stem for p in iter_data_files(out_dir, "*.npz")}
    todo = [p for p in paths
            if Path(p).stem not in on_disk and not ledger.done(str(p))]
    log(f"{len(paths)} granules, {len(todo)} to ingest "
        f"({len(paths) - len(todo)} already cached)")
    log(f"cache -> {Path(cache_root) / cfg.fingerprint()}")
    for i, p in enumerate(todo, 1):
        r = ingest_one(p, cfg, cache_root, delete_raw, strict_scan, write)
        ledger.record(str(p), r)
        if r.state != "cached":
            log(f"  [{i}/{len(todo)}] {Path(p).name}: {r.state} -- {r.error[:90]}")
        if halt_on_scan and r.state == "failed_scan":
            log(f"\n  HALTED on {Path(p).name}: {r.error}")
            log(f"  {i - 1} granule(s) ingested before this. Re-run to resume.")
            for f in r.sentinel_flags:
                if f["suspicion"] == "high":
                    log(f"    {f['channel']} value={f['value']:.4g} "
                        f"{100*f['fraction']:.3f}%  isolation={f['isolation_iqr']:.1f} IQR")
            break
        elif r.high_suspicion:
            log(f"  [{i}/{len(todo)}] {Path(p).name}: cached with "
                f"{r.high_suspicion} HIGH-suspicion pile-up(s)")
        if i % log_every == 0 or i == len(todo):
            log(f"  [{i}/{len(todo)}] {ledger.report().splitlines()[0]}")
    return ledger
