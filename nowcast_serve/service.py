"""The inference service, written the way services are normally written.

That phrasing is the requirement, not a stylistic note. The ordered
workaround in `nowcast_data._threads` says "finish all CPU ingest before you
touch the GPU", which forbids the single most common thing a service does:
build the model and log its device at boot, then handle requests. Under the
split that is legal again --

    NowcastService(...)          # model on the GPU, at boot, first thing
    svc.handle_scan(path)        # ingest in the child, forward in the parent

-- because the resampling happens in a process that has never imported
torch. The ordering constraint is gone, not documented.

The parent installs a guard against pyresample/pykdtree/satpy, so the split
cannot be undone by a later convenience import; it becomes a named
ImportError at the point of the mistake.
"""
from __future__ import annotations

import time

import numpy as np

from .guards import guard_model_process
from .pool import IngestWorker

CHANNELS = ("TIR1", "WV")


class NowcastService:
    """Model in this process, ingest in a child. Timings kept per stage."""

    def __init__(self, model=None, *, device: str | None = None,
                 channels=CHANNELS, dim: int = 128, depth: int = 4,
                 lead_steps: int = 6, context_frames: int = 2,
                 station_coords=None, n_stations: int = 32,
                 guard: bool = True, start_worker: bool = True):
        self.channels = tuple(channels)
        self.timings: dict[str, float] = {}

        # Guard first. `check_absent=False` because a caller may legitimately
        # have imported satpy earlier in a notebook; what matters is that
        # nothing NEW pulls it in behind the model's back.
        self._blocker = guard_model_process(check_absent=False) if guard else None

        # The worker starts BEFORE torch is imported, and on Linux that is
        # load-bearing rather than a latency tweak.
        #
        # Popen cannot use posix_spawn when pass_fds is given (CPython
        # subprocess.py guards on `not pass_fds`), so the child is started
        # with fork+exec. That is SAFE for OpenMP -- the child execs, so the
        # parent's initialised libomp is replaced, which is exactly what
        # multiprocessing's fork never does. But fork from a large-RSS
        # parent is a separate hazard: under strict overcommit
        # (vm.overcommit_memory=2) the kernel reserves the parent's whole
        # commit, and a process holding a model plus a CUDA context is many
        # GB. Forking before the model is loaded keeps that cheap.
        #
        # Correctness does not depend on the order -- the reverse is
        # exercised in the tests -- but memory behaviour does.
        self.worker = IngestWorker().start() if start_worker else None

        import torch
        self.torch = torch
        if device is None:
            device = ("cuda" if torch.cuda.is_available()
                      else "mps" if torch.backends.mps.is_available() else "cpu")
        self.device = device

        t0 = time.perf_counter()
        if model is None:
            from nowcast_model import ModelConfig, MultiTaskNowcaster
            # 2 physical channels + 2 finite-masks
            model = MultiTaskNowcaster(ModelConfig(
                in_channels=2 * len(self.channels), context_frames=context_frames,
                grid_size=96, lead_steps=lead_steps, dim=dim, depth=depth))
        self.model = model.to(self.device).eval()
        self.context_frames = context_frames
        self.timings["model_build"] = time.perf_counter() - t0

        # The cloudburst head is point geometry: it scores at gauge locations,
        # not on the grid, so it needs station coordinates. Real deployment
        # passes the IMD/AWS station list. Anything else is a placeholder and
        # says so in every reply -- an unlabelled placeholder here would put
        # forecasts at invented locations and look entirely normal.
        if station_coords is None:
            self.station_coords = torch.zeros(1, n_stations, 2).uniform_(-1, 1)
            self.stations_are_real = False
        else:
            sc = torch.as_tensor(station_coords, dtype=torch.float32)
            self.station_coords = sc if sc.ndim == 3 else sc[None]
            self.stations_are_real = True
        self.station_coords = self.station_coords.to(self.device)

    # -- request -----------------------------------------------------------
    def handle_scan(self, path, strict: bool = False) -> dict:
        """One inference tick: ingest in the child, forward here."""
        torch = self.torch

        t0 = time.perf_counter()
        arrays, check = self.worker.ingest(path, self.channels, strict=strict)
        self.timings["ingest"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        x = self._assemble(arrays)
        self.timings["assemble"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        with torch.no_grad():
            out = self.model(x, self.station_coords)
        self._sync()
        self.timings["forward"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        probs = {k: torch.sigmoid(v).float().cpu().numpy()
                 for k, v in out.items()}
        self.timings["postprocess"] = time.perf_counter() - t0

        return {"probs": probs, "check": check,
                "timings": dict(self.timings), "device": self.device,
                "stations_are_real": self.stations_are_real}

    def _assemble(self, arrays):
        torch = self.torch
        stack = np.stack([arrays[c] for c in self.channels]).astype(np.float32)
        finite = np.isfinite(stack).astype(np.float32)
        # Crop to the model's grid; the full-grid path is the CUDA probe's job.
        g = self.model.cfg.grid_size if hasattr(self.model, "cfg") else 96
        stack, finite = np.nan_to_num(stack)[:, :g, :g], finite[:, :g, :g]
        both = np.concatenate([stack, finite], axis=0)[None, :, None]
        x = torch.from_numpy(both).repeat(1, 1, self.context_frames, 1, 1)
        return x.to(self.device)

    def _sync(self):
        torch = self.torch
        if self.device == "cuda":
            torch.cuda.synchronize()
        elif self.device == "mps":
            torch.mps.synchronize()

    def close(self):
        if self.worker is not None:
            self.worker.stop()
            self.worker = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
