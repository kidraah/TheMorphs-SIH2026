"""Parent side of the split: a persistent ingest child that is a real exec.

Why not multiprocessing.Process
-------------------------------
The first version of this used `mp.get_context("spawn").Process`. It worked,
and it still had a hole, found by a test written in the service's shape:

    GPU_UP mps        <- the parent
    GPU_UP mps        <- THE INGEST CHILD

spawn re-imports the parent's `__main__` in the child to recover the target
function. So any module-level torch work in the entrypoint -- `import torch`,
a device probe, a device log line, the ordinary first three lines of a
service -- is REPLAYED INSIDE THE INGEST WORKER, putting both OpenMP
runtimes back in one process. The split then held only as long as every
future entrypoint kept its torch work behind `if __name__ == "__main__":`,
which is precisely the "somebody has to remember" property the split exists
to remove.

The same mechanism gives a second failure: a service constructed at module
level starts a worker in every re-imported child, without end.

So the child is launched with `subprocess` as `python -m
nowcast_serve.ingest_worker` -- a fresh interpreter running a named module
that never touches the parent's `__main__`. It cannot replay the
entrypoint, it cannot recurse, and it cannot inherit an initialised libomp
the way fork does. The failure class is gone rather than ordered around.

Measured on this machine, worst case (parent has torch up on MPS, no thread
pinning anywhere):

    fork                     -> child exitcode -6, "OMP: Error #15"
    spawn via Process        -> works, but replays __main__ (above)
    exec'd module (this)     -> works, nothing to replay

Cost of the boundary, same scan, median of 4 after warm-up:

    in-process ingest   2.282 s
    cross-process       2.191 s      (6.30 MB of arrays over the pipe)

i.e. free, and the child start is 0.05 s paid once at boot. Shared memory
would buy lifetime bugs and no measurable time.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import socket
import subprocess
import sys
import time
from multiprocessing.connection import Connection

from .guards import GuardViolation

READY_TIMEOUT_S = 180.0          # exec + satpy import, cold
INGEST_TIMEOUT_S = 300.0
WORKER_MODULE = "nowcast_serve.ingest_worker"


class IngestError(RuntimeError):
    """The worker reported a failure for one scan. The worker is still alive."""


class WorkerDied(RuntimeError):
    """The worker process is gone. Carries the exit code when known."""


class IngestWorker:
    """One exec'd process that decodes and regrids scans.

    Safe to construct in a process that already owns torch and the GPU, from
    module level or anywhere else -- that is the entire point.

        with IngestWorker() as w:
            arrays, check = w.ingest(path)
    """

    def __init__(self, python: str | None = None, guard_child: bool = True):
        self.python = python or sys.executable
        self.guard_child = guard_child
        self._proc: subprocess.Popen | None = None
        self._conn: Connection | None = None
        self.child_pid = None
        self.start_seconds = None

    # -- lifecycle ---------------------------------------------------------
    def start(self, timeout: float = READY_TIMEOUT_S) -> "IngestWorker":
        if self._proc is not None:
            return self

        # socketpair rather than pipes: one duplex channel, and the child
        # keeps stdin/stdout/stderr free for ordinary logging.
        parent_sock, child_sock = socket.socketpair()
        child_fd = child_sock.fileno()
        argv = [self.python, "-m", WORKER_MODULE, "--fd", str(child_fd)]
        if not self.guard_child:
            argv.append("--no-guard")

        env = dict(os.environ)
        env.pop("_NOWCAST_THREADS_PINNED", None)
        # The child owns pyresample's runtime alone, so pinning is belt on
        # top of braces here -- kept because single-threaded resampling is
        # the right default regardless of the conflict.
        env.setdefault("OMP_NUM_THREADS", "1")
        env.setdefault("PYKDTREE_NUM_THREADS", "1")

        t0 = time.perf_counter()
        self._proc = subprocess.Popen(argv, pass_fds=(child_fd,), env=env,
                                      cwd=os.path.dirname(os.path.dirname(
                                          os.path.abspath(__file__))))
        child_sock.close()                     # parent keeps only its end
        self._conn = Connection(parent_sock.detach())

        if not self._conn.poll(timeout):
            self.stop()
            raise WorkerDied(f"ingest worker not ready within {timeout:g}s")
        try:
            hello = self._conn.recv()
        except EOFError:
            code = self._proc.poll()
            self.stop()
            raise WorkerDied(
                f"ingest worker exited during startup (exitcode={code})"
            ) from None
        if not hello.get("ok"):
            self.stop()
            raise WorkerDied(f"ingest worker failed to start: {hello}")
        self.child_pid = hello.get("pid")
        self.start_seconds = time.perf_counter() - t0
        return self

    def stop(self, timeout: float = 10.0) -> None:
        if self._conn is not None:
            try:
                self._conn.send({"op": "stop"})
            except (BrokenPipeError, OSError, ValueError):
                pass
        if self._proc is not None:
            try:
                self._proc.wait(timeout)
            except subprocess.TimeoutExpired:
                self._proc.terminate()
                try:
                    self._proc.wait(5)
                except subprocess.TimeoutExpired:
                    self._proc.kill()
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
        self._proc, self._conn = None, None

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()

    @property
    def alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    @property
    def exitcode(self):
        return self._proc.poll() if self._proc is not None else None

    # -- work --------------------------------------------------------------
    def ingest(self, path, channels=("TIR1", "WV"), strict: bool = False,
               timeout: float = INGEST_TIMEOUT_S):
        """Decode + regrid one scan in the child. Returns (arrays, check).

        `check` is a plain dict, not a ScanCheck: the parent must not have to
        import nowcast_data to read a reply, or the boundary leaks back.
        """
        if not self.alive:
            raise WorkerDied(f"ingest worker is not running "
                             f"(exitcode={self.exitcode}); call start()")
        self._conn.send({"op": "ingest", "path": os.fspath(path),
                         "channels": list(channels), "strict": bool(strict)})
        if not self._conn.poll(timeout):
            raise WorkerDied(f"ingest worker silent for {timeout:g}s "
                             f"(alive={self.alive})")
        try:
            reply = self._conn.recv()
        except EOFError:
            code = self.exitcode
            raise WorkerDied(
                f"ingest worker died during ingest (exitcode={code}). "
                f"-6 is SIGABRT, which for this worker means two OpenMP "
                f"runtimes -- check that nothing imported torch inside it."
            ) from None
        if not reply.get("ok"):
            raise IngestError(reply.get("error", "unknown ingest failure"))
        return reply["arrays"], reply["check"]

    def era5(self, when, timeout: float = INGEST_TIMEOUT_S):
        """Load and regrid the thermodynamic context, also in the child.

        ERA5's regrid is pyresample as well, so putting only the satellite
        decode behind the boundary would leave the conflict in place for
        every run that uses cross-attention context.
        """
        if not self.alive:
            raise WorkerDied(f"ingest worker is not running "
                             f"(exitcode={self.exitcode}); call start()")
        self._conn.send({"op": "era5", "time": str(when)})
        if not self._conn.poll(timeout):
            raise WorkerDied(f"ingest worker silent for {timeout:g}s")
        try:
            reply = self._conn.recv()
        except EOFError:
            raise WorkerDied(f"worker died during era5 load "
                             f"(exitcode={self.exitcode})") from None
        if not reply.get("ok"):
            raise IngestError(reply.get("error", "unknown era5 failure"))
        return reply["fields"], reply["names"]


def default_start_method_is_fork() -> bool:
    """True where CPython would have chosen fork for multiprocessing.

    Kept as a fact this module's design depends on, not as something it
    uses: Lib/multiprocessing/context.py picks spawn on darwin and fork
    everywhere else, so a Process-based design would behave differently on
    the development Mac and the Linux deployment box. The exec'd child here
    is unaffected either way.
    """
    return mp.get_start_method(allow_none=False) == "fork"
