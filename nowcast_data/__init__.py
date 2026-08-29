"""Data ingestion for the nowcasting system.

Importing this package pins OpenMP thread counts (see `_threads`). That
happens at import rather than being left to callers because pykdtree and
torch each ship an OpenMP runtime, and two in one process abort the
interpreter -- which the inference service would do on every request, since
it resamples with pyresample and then infers with torch.

Service entrypoints should still call `pin_threads()` explicitly on their
first line: the variables are read when OpenMP loads, so importing this
package *after* torch is too late.
"""
from ._threads import (already_loaded, ensure_pinned_or_reexec, pin_threads,
                       thread_status)

pin_threads(warn=False)          # warn=False: import order is the caller's job

__all__ = ["pin_threads", "thread_status", "already_loaded",
           "ensure_pinned_or_reexec"]
