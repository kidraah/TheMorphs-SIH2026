"""The ingest child: decode + regrid, in a process that has never seen torch.

Runs a request loop over a Connection. Deliberately dumb -- one job, no
model, no GPU, no device probe. The first thing it does is forbid torch, so
if anyone later adds a convenience `import torch` anywhere under
`nowcast_data`, this process raises a named ImportError on startup instead of
aborting the service under load.
"""
from __future__ import annotations

import os
import traceback

# Order matters even here: guard, then pin, then import the heavy stack.
from .guards import guard_ingest_process


def _serialisable_check(chk) -> dict:
    """ScanCheck carries dataclasses; ship the parts the service acts on.

    Sending a plain dict rather than the object keeps the boundary from
    quietly becoming an import channel -- the parent must not need
    nowcast_data classes to read a reply.
    """
    return {
        "passed": bool(chk.passed),
        "report": chk.report(),
        "metadata": dict(getattr(chk, "metadata", {}) or {}),
        "failures": [str(r) for r in chk.results
                     if not r.passed and not r.skipped],
    }


def serve(conn, *, guard: bool = True) -> None:
    """Request loop. Each request is a dict; each reply is a dict.

    Reply is always {"ok": bool, ...} -- an ingest failure comes back as data,
    because a bad scan must not take the worker down with it.
    """
    if guard:
        guard_ingest_process()

    import nowcast_data  # noqa: F401  (pins OpenMP; still cheap insurance)
    from nowcast_data.insat import ingest_scan

    conn.send({"ok": True, "event": "ready", "pid": os.getpid()})

    while True:
        try:
            req = conn.recv()
        except EOFError:
            return
        if req is None or req.get("op") == "stop":
            return

        try:
            if req["op"] == "era5":
                # ERA5's regrid is pyresample too, so it belongs on this side
                # of the split -- not just the satellite decode.
                from nowcast_data.era5 import load_fields
                from nowcast_data.era5 import resample_to_grid as era5_grid
                f = load_fields(req["time"])
                conn.send({"ok": True, "fields": era5_grid(f),
                           "names": sorted(f.names)})
                continue
            if req["op"] != "ingest":
                raise ValueError(f"unknown op {req['op']!r}")
            channels = tuple(req.get("channels", ("TIR1", "WV")))
            arrays, chk = ingest_scan(req["path"], channels,
                                      strict=bool(req.get("strict", False)))
            conn.send({
                "ok": True,
                "channels": list(channels),
                # float32 (912, 864) is 3.15 MB per channel -- measured
                # cheap enough over the pipe that shared memory would only
                # buy lifetime bugs. See docs/SERVICE_ARCHITECTURE.md.
                "arrays": {c: arrays[c] for c in channels},
                "check": _serialisable_check(chk),
            })
        except Exception as e:
            conn.send({"ok": False, "error": f"{type(e).__name__}: {e}",
                       "traceback": traceback.format_exc()})


def main(conn, guard: bool = True) -> None:
    try:
        serve(conn, guard=guard)
    finally:
        try:
            conn.close()
        except Exception:
            pass


def _cli(argv=None) -> int:
    """`python -m nowcast_serve.ingest_worker --fd N`.

    A module entry point, not a multiprocessing target, and that difference
    is the design. multiprocessing's spawn recovers its target by
    re-importing the parent's __main__ in the child, which replays any
    module-level torch work -- a device probe, a device log line -- inside
    the process that must never see torch. Measured: the parent's "GPU_UP"
    line printed twice, once from the ingest worker.

    Exec'ing a named module has no __main__ to replay.
    """
    import argparse
    from multiprocessing.connection import Connection

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fd", type=int, required=True,
                    help="inherited socketpair fd to the parent")
    ap.add_argument("--no-guard", action="store_true",
                    help="skip the torch import guard (tests only)")
    a = ap.parse_args(argv)

    conn = Connection(a.fd)
    main(conn, guard=not a.no_guard)
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
