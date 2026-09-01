"""Where raw staging and the decoded cache live. Configured before the pull.

Sizing, from the corrected archive numbers (LIMITATIONS 12-13):

    raw staging   a few GB at a time -- decode-and-discard means raw NEVER
                  accumulates. If it grows past ~10 GB the decode step has
                  stopped and the pull should be halted.
    decoded cache ~303 GB at 19,200 scans x 15.8 MB (six channels, the
                  antecedent channel, and finite masks, uint16).

Both default to the external volume. The cache does not fit alongside ~120 GB
of internal free space, and training reads it from a network volume on the
rented box anyway, so local read speed is a development concern rather than a
training one.

Override with NOWCAST_RAW_DIR and NOWCAST_CACHE_DIR.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

DEFAULT_ROOT = Path("/Volumes/Untitled/claude_down")

# If raw staging exceeds this, decode-and-discard has stopped running and the
# pull is on its way to 8.2 TB instead of a few GB.
RAW_ALARM_GB = 10.0
EXPECTED_CACHE_GB = 303.0


def raw_dir() -> Path:
    return Path(os.environ.get("NOWCAST_RAW_DIR", DEFAULT_ROOT / "raw"))


def cache_dir() -> Path:
    return Path(os.environ.get("NOWCAST_CACHE_DIR", DEFAULT_ROOT / "cache"))


def _free_gb(p: Path) -> float:
    probe = p
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    return shutil.disk_usage(probe).free / 1024 ** 3


def dir_size_gb(p: Path) -> float:
    if not p.exists():
        return 0.0
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1024 ** 3


def check(create: bool = False) -> dict:
    """Report both paths, their free space, and whether raw is accumulating.

    Called before a pull so the answer is known in advance rather than when
    the volume fills mid-transfer.
    """
    raw, cache = raw_dir(), cache_dir()
    if create:
        raw.mkdir(parents=True, exist_ok=True)
        cache.mkdir(parents=True, exist_ok=True)

    raw_used = dir_size_gb(raw)
    out = {
        "raw_dir": str(raw), "cache_dir": str(cache),
        "raw_exists": raw.exists(), "cache_exists": cache.exists(),
        "raw_used_gb": raw_used,
        "cache_used_gb": dir_size_gb(cache),
        "raw_free_gb": _free_gb(raw), "cache_free_gb": _free_gb(cache),
        "warnings": [],
    }
    if raw_used > RAW_ALARM_GB:
        out["warnings"].append(
            f"raw staging holds {raw_used:.1f} GB, above the {RAW_ALARM_GB:.0f} GB "
            f"alarm. Decode-and-discard is not running: raw should never "
            f"accumulate, and the full pull is 8.2 TB.")
    if out["cache_free_gb"] < EXPECTED_CACHE_GB:
        out["warnings"].append(
            f"cache volume has {out['cache_free_gb']:.0f} GB free, below the "
            f"~{EXPECTED_CACHE_GB:.0f} GB the full decoded cache needs.")
    return out


def report() -> str:
    c = check()
    lines = [f"  raw staging : {c['raw_dir']}",
             f"                {c['raw_used_gb']:.1f} GB used, "
             f"{c['raw_free_gb']:.0f} GB free"
             + ("" if c["raw_exists"] else "   [does not exist yet]"),
             f"  decoded cache: {c['cache_dir']}",
             f"                {c['cache_used_gb']:.1f} GB used, "
             f"{c['cache_free_gb']:.0f} GB free"
             + ("" if c["cache_exists"] else "   [does not exist yet]")]
    for w in c["warnings"]:
        lines.append(f"  !! {w}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# File walking. Every site must go through this.
# ---------------------------------------------------------------------------
#
# macOS writes an AppleDouble sidecar "._name" beside every file on a
# non-HFS volume, and an unfiltered walk counts them as data. Three separate
# incidents from this one cause:
#
#   1. "352 npz" reported for a directory holding 176
#   2. a migration failure counter reading 176/176, because every sidecar
#      failed to parse as an npz
#   3. the cache-root guard reporting 1,562 files for 781, exactly 2x --
#      and a guard that reports double the truth is one you start
#      discounting, which is worse than a guard that is merely absent
#
# Fixed here rather than at each call site, because "remember to filter" is
# the class of instruction this project has repeatedly failed to keep.

def is_sidecar(p) -> bool:
    """macOS AppleDouble resource fork, not data."""
    return Path(p).name.startswith("._")


def iter_data_files(root, pattern: str = "*", recursive: bool = True):
    """Every real file under `root` matching `pattern`, sidecars excluded."""
    root = Path(root)
    if not root.exists():
        return
    it = root.rglob(pattern) if recursive else root.glob(pattern)
    for p in it:
        if p.is_file() and not is_sidecar(p):
            yield p


def count_data_files(root, pattern: str = "*") -> int:
    return sum(1 for _ in iter_data_files(root, pattern))
