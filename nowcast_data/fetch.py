"""Resumable, verifying fetcher. Built before the pull, not during it.

The measured failure rate makes this a requirement, not a nicety: the 10.5 h
event pull produced one HTTP 500 and left **seven `.part` files**. Over a
5.3-day archive pull at the measured 7.9 MB/s, failures are certain.

Four properties, and the first is the one that fixes the seven .part files:

1. **A file at its final name is always complete.** Downloads go to `.part`,
   are verified, and only then `os.replace`d into place. A crash can leave a
   `.part`; it can never leave a truncated file that looks finished.
2. **Verified before being marked complete.** Size alone is not verification
   -- a truncated HDF5 with the right Content-Length is impossible, but a
   *complete* HDF5 that is an HTML error page is not, and that is instance 6.
   So verification opens the file and requires the datasets to be there.
3. **Resume from partial**, when the server honours Range. When it does not,
   restart -- and say so, rather than appending to a `.part` and producing a
   file that is the right length and corrupt.
4. **Identical behaviour either way.** Range changes how much is transferred
   and whether a partial can be resumed. It changes nothing about what
   counts as complete, so the manifest and the verification are the same.

The manifest is the operator's view: what is done, what failed, why, and how
many attempts each took.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

PENDING, COMPLETE, FAILED = "pending", "complete", "failed"


@dataclass
class Item:
    """One file to fetch."""
    url: str
    dest: str
    identifier: str = ""
    state: str = PENDING
    bytes_written: int = 0
    attempts: int = 0
    error: str = ""
    sha256: str = ""
    seconds: float = 0.0
    mode: str = ""            # "whole" or "datasets"

    def key(self) -> str:
        return self.identifier or self.dest


class VerificationError(RuntimeError):
    """The bytes arrived and are not what was asked for."""


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify_hdf5(path, required=(), min_bytes: int = 1 << 20) -> None:
    """Open it and require the datasets. Raises VerificationError.

    Size is checked first only to give a better message: the real test is
    that h5py can open it and the datasets are present. A login page, an
    error page or a relocation stub all have a plausible size and none of
    them open as HDF5 -- which is the whole lesson of instance 6.
    """
    p = Path(path)
    if not p.exists():
        raise VerificationError(f"{p} does not exist")
    size = p.stat().st_size
    if size < min_bytes:
        head = p.open("rb").read(200)
        raise VerificationError(
            f"{p.name} is only {size} bytes -- probably an error or login "
            f"page, not a scan. First bytes: {head[:80]!r}")
    try:
        import h5py
        with h5py.File(p, "r") as f:
            missing = [n for n in required if n not in f]
    except VerificationError:
        raise
    except Exception as e:
        raise VerificationError(
            f"{p.name} does not open as HDF5 ({type(e).__name__}: {e}). "
            f"A complete-looking file that is not HDF5 is an error page.") from None
    if missing:
        raise VerificationError(f"{p.name} is missing datasets: {missing}")


def sha256_of(path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

class Manifest:
    """Append-safe JSON state. Written after every item, so a kill -9 costs
    at most one file's progress."""

    def __init__(self, path):
        self.path = Path(path)
        self.items: dict[str, Item] = {}
        if self.path.exists():
            raw = json.loads(self.path.read_text())
            self.items = {k: Item(**v) for k, v in raw.get("items", {}).items()}

    def add(self, item: Item) -> Item:
        return self.items.setdefault(item.key(), item)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(
            {"updated": time.time(),
             "items": {k: asdict(v) for k, v in self.items.items()}},
            indent=1))
        os.replace(tmp, self.path)          # never a half-written manifest

    def counts(self) -> dict:
        c = {PENDING: 0, COMPLETE: 0, FAILED: 0}
        for i in self.items.values():
            c[i.state] = c.get(i.state, 0) + 1
        return c

    def failures(self) -> list[Item]:
        return [i for i in self.items.values() if i.state == FAILED]

    def report(self) -> str:
        c = self.counts()
        done = sum(i.bytes_written for i in self.items.values()
                   if i.state == COMPLETE)
        secs = sum(i.seconds for i in self.items.values())
        rate = done / secs / 1e6 if secs else float("nan")
        lines = [f"{c[COMPLETE]} complete, {c[PENDING]} pending, "
                 f"{c[FAILED]} failed of {len(self.items)}",
                 f"    {done/1024**3:.1f} GB written, {rate:.1f} MB/s mean"]
        for i in self.failures()[:10]:
            lines.append(f"    FAILED {i.key()}: {i.error[:110]}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Fetcher
# ---------------------------------------------------------------------------

@dataclass
class FetchConfig:
    max_attempts: int = 5
    backoff_base_s: float = 5.0
    backoff_cap_s: float = 300.0
    chunk: int = 1 << 20
    timeout_s: float = 300.0
    required_datasets: tuple = ()
    keep_datasets: tuple = ()      # non-empty -> dataset mode when Range works
    checksum: bool = False         # sha256 costs a full re-read; off by default
    min_bytes: int = 1 << 20


class Fetcher:
    def __init__(self, manifest_path, config: FetchConfig | None = None,
                 session=None, log=print):
        self.cfg = config or FetchConfig()
        self.manifest = Manifest(manifest_path)
        self.session = session
        self.log = log
        self._range_ok: dict[str, bool] = {}

    # -- range support, cached per host ------------------------------------
    def range_supported(self, url) -> bool:
        host = url.split("/")[2] if "//" in url else url
        if host not in self._range_ok:
            from .ranged import probe_range_support
            r = probe_range_support(url, self.session)
            self._range_ok[host] = r.supported
            self.log(f"  range support for {host}: {r.supported} ({r.reason})")
        return self._range_ok[host]

    # -- one file -----------------------------------------------------------
    def fetch_one(self, item: Item) -> Item:
        dest = Path(item.dest)
        part = dest.with_suffix(dest.suffix + ".part")

        # Already done? Verify rather than trust the manifest: the file may
        # have been deleted or truncated since it was written.
        if item.state == COMPLETE and dest.exists():
            try:
                verify_hdf5(dest, self.cfg.required_datasets, self.cfg.min_bytes)
                return item
            except VerificationError as e:
                self.log(f"  re-fetching {dest.name}: {e}")
                item.state = PENDING

        for attempt in range(item.attempts, self.cfg.max_attempts):
            item.attempts = attempt + 1
            t0 = time.perf_counter()
            try:
                if self.cfg.keep_datasets and self.range_supported(item.url):
                    self._fetch_datasets(item, part)
                    item.mode = "datasets"
                else:
                    self._fetch_whole(item, part)
                    item.mode = "whole"
                verify_hdf5(part, self.cfg.required_datasets, self.cfg.min_bytes)
                if self.cfg.checksum:
                    item.sha256 = sha256_of(part)
                dest.parent.mkdir(parents=True, exist_ok=True)
                os.replace(part, dest)      # atomic: final name == complete
                item.bytes_written = dest.stat().st_size
                item.seconds += time.perf_counter() - t0
                item.state, item.error = COMPLETE, ""
                return item
            except Exception as e:
                item.seconds += time.perf_counter() - t0
                item.error = f"{type(e).__name__}: {e}"
                if isinstance(e, VerificationError) and part.exists():
                    # Bad bytes must not be resumed onto -- appending to a
                    # corrupt partial yields a file of the right length that
                    # is wrong, which is worse than starting over.
                    part.unlink()
                if item.attempts >= self.cfg.max_attempts:
                    item.state = FAILED
                    return item
                delay = min(self.cfg.backoff_cap_s,
                            self.cfg.backoff_base_s * 2 ** attempt)
                delay *= 0.5 + random.random()      # jitter; a synchronised
                self.log(f"  attempt {item.attempts} failed for {item.key()}: "
                         f"{item.error[:90]} -- retrying in {delay:.0f}s")
                time.sleep(delay)
        item.state = FAILED
        return item

    def _fetch_whole(self, item: Item, part: Path) -> None:
        import requests

        s = self.session or requests.Session()
        part.parent.mkdir(parents=True, exist_ok=True)
        start = part.stat().st_size if part.exists() else 0
        headers = {}
        if start and self.range_supported(item.url):
            headers["Range"] = f"bytes={start}-"
        elif start:
            # Cannot resume -- restart rather than append blindly.
            self.log(f"  {item.key()}: server will not resume, restarting "
                     f"({start/1e6:.0f} MB discarded)")
            part.unlink()
            start = 0

        r = s.get(item.url, headers=headers, stream=True,
                  timeout=self.cfg.timeout_s)
        r.raise_for_status()
        if start and r.status_code != 206:
            # Asked to resume, got the whole body: truncate and take it, or
            # we would concatenate the file onto its own tail.
            part.unlink()
            start = 0
        with open(part, "ab" if start else "wb") as fh:
            for chunk in r.iter_content(self.cfg.chunk):
                fh.write(chunk)

    def _fetch_datasets(self, item: Item, part: Path) -> None:
        """Write a small HDF5 containing only the kept datasets."""
        import h5py

        from .ranged import read_datasets

        got = read_datasets(item.url, list(self.cfg.keep_datasets), self.session)
        if not got.datasets:
            raise VerificationError(f"no datasets returned for {item.key()}")
        part.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(part, "w") as out:
            for name, arr in got.datasets.items():
                out.create_dataset(name, data=arr, compression="gzip",
                                   compression_opts=1)
            out.attrs["source_url"] = item.url
            out.attrs["ranged_bytes_transferred"] = got.bytes_transferred
            out.attrs["subset_of_full_scan"] = True

    # -- the pull -----------------------------------------------------------
    def fetch_all(self, items, progress_every: int = 25) -> Manifest:
        items = [self.manifest.add(i) for i in items]
        # A COMPLETE item whose file is gone is NOT complete. Filtering on
        # manifest state alone is trusting the manifest over the filesystem,
        # which is how a pull "finishes" with files missing -- found by the
        # test written for exactly this, which the first version failed.
        todo = [i for i in items
                if i.state != COMPLETE or not Path(i.dest).exists()]
        self.log(f"{len(items)} items, {len(todo)} to fetch "
                 f"({len(items)-len(todo)} already complete)")
        for n, item in enumerate(todo, 1):
            self.fetch_one(item)
            self.manifest.save()            # after EVERY item
            if n % progress_every == 0 or n == len(todo):
                self.log(f"[{n}/{len(todo)}] {self.manifest.report().splitlines()[0]}")
        return self.manifest


def sweep_orphaned_parts(root, older_than_s: float = 3600.0) -> list[str]:
    """Report `.part` files with no active writer. The seven left behind by
    the event pull are exactly this."""
    now = time.time()
    out = []
    for p in Path(root).rglob("*.part"):
        if now - p.stat().st_mtime > older_than_s:
            out.append(str(p))
    return out
