"""Inference service: torch and pyresample in separate processes.

The OpenMP conflict between pykdtree (pyresample/satpy) and torch used to be
handled by ordering -- pin the thread variables, finish CPU ingest before
touching the GPU. `nowcast_serve` removes the constraint instead of ordering
around it: the model process never imports pyresample, the ingest process
never imports torch, and the boundary is a spawned interpreter.

    from nowcast_serve import NowcastService
    with NowcastService() as svc:          # model on the GPU, at boot
        result = svc.handle_scan(path)     # ingest in the child

See docs/SERVICE_ARCHITECTURE.md.
"""
from .guards import (GuardViolation, guard_ingest_process, guard_model_process)
from .pool import IngestError, IngestWorker, WorkerDied
from .service import NowcastService

__all__ = ["NowcastService", "IngestWorker", "IngestError", "WorkerDied",
           "GuardViolation", "guard_ingest_process", "guard_model_process"]
