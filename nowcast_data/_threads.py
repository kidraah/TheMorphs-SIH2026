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

IN-PROCESS PINNING IS BEST-EFFORT AND NOT SUFFICIENT
----------------------------------------------------
Measured: setting these from Python via os.environ works for some call
paths and NOT for others. scripts/probe_latency.py aborts with
"OMP: Error #15" despite pin_threads() having set KMP_DUPLICATE_LIB_OK,
and the same script runs clean when the variable is exported in the shell
first. The test suite passes because it exercises a different resample path
than the real ingest does -- a test that passes while production aborts,
which is the exact gap this module was written to close.

So: `ensure_pinned_or_reexec()` is the reliable form. It re-executes the
interpreter with the variables set when they were absent at startup, which
is the only way to guarantee OpenMP sees them. Entrypoints should call it
on their first line.
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


def ensure_pinned_or_reexec(argv=None) -> None:
    """Guarantee the OpenMP variables are set, re-executing if necessary.

    Setting os.environ from Python is too late for some libraries: they read
    the variables when their runtime loads, and by the time any Python code
    runs, a linked libomp may already be initialised. The only reliable fix
    is for the variables to exist before the interpreter starts.

    So if they were missing at startup, set them and re-exec this same
    command. The re-exec happens once, costs an interpreter restart, and is
    invisible to the caller.

    Call it on the FIRST line of an entrypoint, before any other import:

        from nowcast_data._threads import ensure_pinned_or_reexec
        ensure_pinned_or_reexec()

    No-op if NOWCAST_NO_THREAD_PIN is set, or if already re-executed (guarded
    by a sentinel variable so it cannot loop).
    """
    import sys

    if os.environ.get("NOWCAST_NO_THREAD_PIN"):
        return
    if os.environ.get("_NOWCAST_THREADS_PINNED"):
        return

    missing = [k for k in PINNED if k not in os.environ]
    if not missing:
        os.environ["_NOWCAST_THREADS_PINNED"] = "1"
        return

    env = dict(os.environ)
    env.update(PINNED)
    env["_NOWCAST_THREADS_PINNED"] = "1"
    os.execve(sys.executable, [sys.executable] + list(argv or sys.argv), env)
