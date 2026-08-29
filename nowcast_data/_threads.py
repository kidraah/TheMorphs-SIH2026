"""OpenMP thread pinning. Must run before pyresample or torch import.

The hazard
----------
pykdtree (under pyresample's nearest-neighbour resampling) and torch each
ship their own OpenMP runtime. Two OpenMP runtimes in one process abort the
interpreter -- not an exception, a `Fatal Python error: Aborted`.

This was first hit in the test suite, where every real-file test passed
alone and the suite crashed in company. **It is not a test problem.** The
inference service does exactly the same thing in one process: pyresample to
put an INSAT scan on the analysis grid, then torch to run the model. Pinning
it only in conftest.py would fix CI and leave production to abort under load.

The catch
---------
OpenMP reads these variables when its runtime loads, so setting them later
has no effect. `pin_threads()` therefore has to run before the first import
of torch or pyresample -- which is why it is called from this package's
__init__, and why service entrypoints should call it explicitly on the first
line, before any other project import.

`already_loaded()` reports when it is too late, rather than silently doing
nothing.
"""
from __future__ import annotations

import os
import sys
import warnings

# Single-threaded is the right default here regardless of the conflict: the
# resampling is one KD-tree query per scan and the model is the thing that
# should own the cores.
PINNED = {
    "OMP_NUM_THREADS": "1",
    "PYKDTREE_NUM_THREADS": "1",
    "KMP_DUPLICATE_LIB_OK": "TRUE",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
}

AT_RISK = ("torch", "pykdtree", "pyresample")


def already_loaded() -> list[str]:
    """Modules whose OpenMP runtime may already be initialised."""
    return [m for m in AT_RISK if m in sys.modules]


def pin_threads(force: bool = False, warn: bool = True) -> dict:
    """Set the OpenMP variables. Returns what was applied.

    Existing values are respected unless `force` -- an operator who has
    deliberately set OMP_NUM_THREADS=8 should not be silently overridden.
    Opt out entirely with NOWCAST_NO_THREAD_PIN=1.
    """
    if os.environ.get("NOWCAST_NO_THREAD_PIN"):
        return {}

    late = already_loaded()
    if late and warn:
        warnings.warn(
            f"pin_threads() called after {late} were imported -- OpenMP reads "
            f"these variables at load time, so this may have no effect. Call it "
            f"on the first line of the entrypoint, before any other import.",
            RuntimeWarning, stacklevel=2)

    applied = {}
    for k, v in PINNED.items():
        if force or k not in os.environ:
            os.environ[k] = v
            applied[k] = v
    return applied


def thread_status() -> dict:
    """Current values, for logging at service start."""
    return {k: os.environ.get(k) for k in PINNED} | {
        "already_loaded": already_loaded(),
        "opted_out": bool(os.environ.get("NOWCAST_NO_THREAD_PIN")),
    }
