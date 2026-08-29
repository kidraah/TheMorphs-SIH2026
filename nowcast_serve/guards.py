"""Import guards that make the OpenMP split structural instead of remembered.

The constraint being removed
----------------------------
pykdtree (under pyresample/satpy) and torch each ship an OpenMP runtime, and
two initialised in one process abort the interpreter -- SIGABRT, not an
exception. `nowcast_data._threads` ORDERS around it: pin the variables, do
CPU ingest before touching the GPU. That works and it is fragile, because it
asks every future entrypoint, wrapper and scheduler to preserve an ordering
that nothing enforces.

This module supports the structural fix instead: the two runtimes never live
in the same process. The parent owns torch; a spawned child owns resampling.
The guards below make a violation an immediate, named ImportError at the
moment someone writes it, rather than a SIGABRT under load six months later.

Measured, on this machine, parent with torch already up on MPS and NO thread
pinning at all:

    spawn -> child exitcode 0, ingest returns (912, 864)
    fork  -> child exitcode -6, "OMP: Error #15 ... already initialized"

So the boundary only removes the constraint if the child is a FRESH
interpreter. fork inherits the parent's already-initialised libomp and dies.
That matters more than it looks: CPython picks the default by platform
(Lib/multiprocessing/context.py) --

    if sys.platform == 'darwin':  spawn
    else:                         fork

-- so plain `multiprocessing.Process(...)` works on a macOS laptop and
aborts on the Linux box it deploys to. `nowcast_serve.pool` therefore names
the spawn context explicitly and refuses fork.
"""
from __future__ import annotations

import sys

# The child must never pull these in.
TORCH_SIDE = ("torch",)
# The parent must never pull these in.
RESAMPLE_SIDE = ("pyresample", "pykdtree", "satpy")


class GuardViolation(ImportError):
    """A process imported a module belonging to the other side of the split."""


class _Blocker:
    """A sys.meta_path finder that refuses named modules.

    Sits at the front of meta_path, so it is consulted before any real
    finder and blocks the import before the extension module -- and its
    OpenMP runtime -- is ever loaded. Blocking after the fact would be
    useless: the abort happens at load time.
    """

    def __init__(self, names: tuple[str, ...], why: str):
        self.names = tuple(names)
        self.why = why

    def _blocked(self, fullname: str) -> str | None:
        for n in self.names:
            if fullname == n or fullname.startswith(n + "."):
                return n
        return None

    def find_module(self, fullname, path=None):        # legacy API, harmless
        return None

    def find_spec(self, fullname, path=None, target=None):
        hit = self._blocked(fullname)
        if hit is None:
            return None
        raise GuardViolation(
            f"{fullname!r} must not be imported in this process: {self.why}\n"
            f"Blocked module family: {hit!r}. This guard exists because the "
            f"failure it prevents is a SIGABRT from two OpenMP runtimes, not "
            f"an exception -- so it cannot be caught where it happens."
        )


def assert_absent(names, side: str) -> None:
    """Fail if a forbidden module is ALREADY imported. Call before guarding."""
    late = [n for n in names if n in sys.modules]
    if late:
        raise GuardViolation(
            f"{late} already imported in the {side} process. The OpenMP "
            f"runtime loads with the module, so guarding now is too late -- "
            f"this import has to be removed, not deferred.")


def forbid(names, why: str, check_absent: bool = True) -> _Blocker:
    """Install a meta_path blocker for `names`. Returns it, for removal."""
    if check_absent:
        assert_absent(names, why)
    blocker = _Blocker(names, why)
    sys.meta_path.insert(0, blocker)
    return blocker


def release(blocker: _Blocker) -> None:
    """Remove a blocker (tests, and nothing else)."""
    try:
        sys.meta_path.remove(blocker)
    except ValueError:
        pass


def guard_ingest_process() -> _Blocker:
    """Called by the ingest child. Nothing here may touch torch."""
    return forbid(TORCH_SIDE,
                  "this is the ingest worker -- it owns pyresample's OpenMP "
                  "runtime, and torch would load a second one")


def guard_model_process(check_absent: bool = True) -> _Blocker:
    """Called by the parent/service. Resampling belongs to the child."""
    return forbid(RESAMPLE_SIDE,
                  "this is the model process -- it owns torch's OpenMP "
                  "runtime, and pyresample would load a second one",
                  check_absent=check_absent)
