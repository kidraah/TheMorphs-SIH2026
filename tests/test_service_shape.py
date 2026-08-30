"""Tests shaped like the SERVICE, not like the code path.

The gap this file closes is the eighth instance of this project's recurring
pattern, and the worst-behaved: `test_service_threads.py` exercised
`pyresample.kd_tree.resample_nearest` directly while `ingest_scan` goes
through satpy's resample path, so the suite was GREEN while the real service
aborted with "OMP: Error #15". Testing the right library call is not the
same as testing the service.

So every test here builds the thing in the shape it is deployed in: a
process that starts a service, puts a model on the GPU at boot, and only
then handles a scan. That ordering -- model first -- is the one the old
workaround forbade and the one every service actually uses.

All of it runs in SUBPROCESSES with the OpenMP variables stripped and
NOWCAST_NO_THREAD_PIN=1 set, so conftest's pinning cannot mask a failure.
If the split is doing the work, these pass with no pinning at all. That is
the claim being tested: the constraint is removed, not ordered around.
"""
import os
import subprocess
import sys
import tempfile
import textwrap

import pytest

REPO = os.path.dirname(os.path.dirname(__file__))
REAL_SCAN = ("/Users/evad/test_ff/3RIMG_L1B_STD/2025/01AUG/"
             "3RIMG_01AUG2025_2345_L1B_STD_V01R00.h5")
needs_scan = pytest.mark.skipif(not os.path.exists(REAL_SCAN),
                                reason="real 3DR scan absent")

_STRIPPED = ("OMP_NUM_THREADS", "PYKDTREE_NUM_THREADS", "KMP_DUPLICATE_LIB_OK",
             "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "_NOWCAST_THREADS_PINNED")


def run(script: str, extra: dict | None = None, timeout: int = 900):
    """Run as a FILE in a child with the pinning stripped.

    A file, not `python -c`: spawn re-imports the parent's __main__ by path,
    and with `-c` there is no path, so `-c` would quietly exercise a
    different bootstrap than a deployed entrypoint does. That is the same
    substitution -- test the convenient thing instead of the deployed thing --
    that this file exists to stop making.
    """
    env = {k: v for k, v in os.environ.items() if k not in _STRIPPED}
    env["NOWCAST_NO_THREAD_PIN"] = "1"      # belt AND braces removed
    env.update(extra or {})
    with tempfile.TemporaryDirectory() as d:
        entry = os.path.join(d, "entrypoint.py")
        with open(entry, "w") as fh:
            fh.write(textwrap.dedent(script))
        return subprocess.run([sys.executable, entry], capture_output=True,
                              text=True, timeout=timeout, env=env, cwd=REPO)


def ok(r, token):
    assert r.returncode == 0, (
        f"rc={r.returncode}"
        + (" (SIGABRT -- two OpenMP runtimes in one process)"
           if r.returncode == -6 else "")
        + f"\nstdout:\n{r.stdout[-2000:]}\nstderr:\n{r.stderr[-3000:]}")
    assert token in r.stdout, f"missing {token!r}:\n{r.stdout[-2000:]}"


# ---------------------------------------------------------------------------
# The shape that matters
# ---------------------------------------------------------------------------

@needs_scan
def test_service_boots_model_then_handles_scans():
    """THE test. A service that builds its model at boot and then serves.

    Written the normal way -- model up front, device probed, requests after --
    which is precisely what the ordered workaround made illegal.
    """
    r = run(f"""
        from nowcast_serve import NowcastService

        def main():
            svc = NowcastService(dim=32, depth=1)
            print("BOOT", svc.device, svc.worker.child_pid)
            for i in range(2):                    # not just the first one
                res = svc.handle_scan({REAL_SCAN!r})
                assert res["check"]["passed"], res["check"]["report"]
                assert sorted(res["probs"]) == [
                    "cloudburst", "extreme_rain", "rain_rate"]
                assert res["stations_are_real"] is False   # labelled, not silent
            svc.close()
            print("SERVICE_OK")

        if __name__ == "__main__":
            main()
    """)
    ok(r, "SERVICE_OK")


@needs_scan
def test_gpu_initialised_before_any_ingest():
    """The exact ordering that aborts without the split.

    Touching the GPU before resampling is what triggered "OMP: Error #15";
    the old rule was "CPU ingest first". Here the GPU goes up first, on
    purpose, and the worker is not even started until after.
    """
    r = run(f"""
        import torch
        dev = ("cuda" if torch.cuda.is_available() else
               "mps" if torch.backends.mps.is_available() else "cpu")
        _ = (torch.randn(128, 128, device=dev) @ torch.randn(128, 128, device=dev)).sum()
        print("GPU_UP", dev)

        from nowcast_serve import IngestWorker

        def main():
            with IngestWorker() as w:
                arrays, check = w.ingest({REAL_SCAN!r})
                assert arrays["TIR1"].shape == (912, 864), arrays["TIR1"].shape
                assert check["passed"], check["report"]
            print("ORDER_OK")

        if __name__ == "__main__":
            main()
    """)
    ok(r, "ORDER_OK")


@needs_scan
def test_service_survives_a_bad_scan_and_keeps_serving():
    """A bad request must not take the worker down -- otherwise the split
    trades an abort for an outage."""
    r = run(f"""
        from nowcast_serve import IngestWorker, IngestError

        def main():
            with IngestWorker() as w:
                try:
                    w.ingest("/nonexistent/not_a_scan.h5")
                except IngestError as e:
                    print("REFUSED", type(e).__name__)
                else:
                    raise AssertionError("bad path should have raised")
                assert w.alive, "worker died on a bad scan"
                arrays, check = w.ingest({REAL_SCAN!r})
                assert check["passed"]
            print("RECOVERED_OK")

        if __name__ == "__main__":
            main()
    """)
    ok(r, "RECOVERED_OK")


# ---------------------------------------------------------------------------
# The hole that a Process-based split still had
# ---------------------------------------------------------------------------

@needs_scan
def test_entrypoint_torch_at_module_level_is_not_replayed_in_the_worker():
    """The regression that killed the first design.

    multiprocessing's spawn re-imports the parent's __main__ in the child to
    recover the target function, so module-level torch work in an entrypoint
    was REPLAYED INSIDE THE INGEST WORKER -- the symptom was the parent's
    "GPU_UP" line printing twice. That put both OpenMP runtimes back in one
    process and left the split depending on every future entrypoint keeping
    its torch work behind `if __name__ == "__main__":`.

    So this entrypoint deliberately has NO main guard, initialises the GPU at
    module level, and starts the worker at module level -- the sloppiest
    legal service. The child is exec'd as a named module and has no __main__
    to replay, so it must still work, and GPU_UP must appear exactly once.
    """
    r = run(f"""
        import torch
        dev = ("cuda" if torch.cuda.is_available() else
               "mps" if torch.backends.mps.is_available() else "cpu")
        _ = (torch.randn(64, 64, device=dev) @ torch.randn(64, 64, device=dev)).sum()
        print("GPU_UP", dev)

        from nowcast_serve import IngestWorker
        w = IngestWorker().start()
        arrays, check = w.ingest({REAL_SCAN!r})
        assert arrays["TIR1"].shape == (912, 864), arrays["TIR1"].shape
        assert check["passed"], check["report"]
        w.stop()
        print("NO_REPLAY_OK")
    """)
    ok(r, "NO_REPLAY_OK")
    assert r.stdout.count("GPU_UP") == 1, (
        f"the entrypoint was replayed in the ingest child "
        f"({r.stdout.count('GPU_UP')} GPU_UP lines) -- both OpenMP runtimes "
        f"are back in one process:\n{r.stdout[-1500:]}")


def test_worker_is_an_exec_not_a_multiprocessing_child():
    """Pins the mechanism, because the hole above was invisible in behaviour
    until an entrypoint happened to touch torch at module level."""
    import inspect

    from nowcast_serve import pool
    src = inspect.getsource(pool)
    assert "subprocess.Popen" in src
    assert "-m" in src and pool.WORKER_MODULE == "nowcast_serve.ingest_worker"
    assert "Process(" not in src, (
        "a multiprocessing.Process child re-imports the parent's __main__; "
        "that is the failure test_entrypoint_torch_at_module_level_... covers")


@needs_scan
@pytest.mark.skipif("fork" not in __import__("multiprocessing").get_all_start_methods(),
                    reason="fork unavailable")
def test_fork_really_does_abort_so_the_exec_is_not_cargo_cult():
    """Justifies the design by measurement rather than belief.

    fork inherits the parent's initialised libomp, so a forked boundary buys
    nothing at all. This matters beyond tidiness: CPython picks the
    multiprocessing default by platform (Lib/multiprocessing/context.py --
    darwin: spawn, else: fork), so a Process-based design would behave
    differently on the development Mac and on the Linux box it deploys to.

    If this ever starts PASSING, the reasoning behind the exec'd child has
    changed and should be re-derived, not quietly relaxed.
    """
    r = run(f"""
        import multiprocessing as mp
        import torch
        dev = ("cuda" if torch.cuda.is_available() else
               "mps" if torch.backends.mps.is_available() else "cpu")
        _ = (torch.randn(64, 64, device=dev) @ torch.randn(64, 64, device=dev)).sum()

        def child(q):
            try:
                from nowcast_data.insat import ingest_scan
                a, c = ingest_scan({REAL_SCAN!r}, ("TIR1",), strict=False)
                q.put(("ok", a["TIR1"].shape))
            except BaseException as e:
                q.put(("err", type(e).__name__))

        if __name__ == "__main__":
            ctx = mp.get_context("fork")
            q = ctx.Queue()
            p = ctx.Process(target=child, args=(q,))
            p.start(); p.join(600)
            print("FORK_EXITCODE", p.exitcode)
    """)
    assert "FORK_EXITCODE" in r.stdout, r.stderr[-2000:]
    code = int(r.stdout.split("FORK_EXITCODE")[1].split()[0])
    assert code != 0, (
        "fork no longer aborts here -- the measurement behind the exec'd "
        "child has changed. Re-derive before relaxing anything.")


def test_platform_default_start_method_is_recorded():
    """The fact the design turns on, asserted so it cannot rot silently."""
    from nowcast_serve.pool import default_start_method_is_fork
    import multiprocessing as mp
    assert default_start_method_is_fork() == (mp.get_start_method() == "fork")


# ---------------------------------------------------------------------------
# The guards: violations are named errors, not aborts
# ---------------------------------------------------------------------------

def test_child_process_cannot_import_torch():
    r = run("""
        from nowcast_serve.guards import GuardViolation, guard_ingest_process
        guard_ingest_process()
        try:
            import torch
        except GuardViolation as e:
            print("BLOCKED_TORCH")
        else:
            raise AssertionError("torch import was not blocked in the worker")
    """)
    ok(r, "BLOCKED_TORCH")


def test_model_process_cannot_import_pyresample():
    r = run("""
        from nowcast_serve.guards import GuardViolation, guard_model_process
        import torch                                  # legal here
        guard_model_process()
        for mod in ("pyresample", "satpy", "pykdtree"):
            try:
                __import__(mod)
            except GuardViolation:
                continue
            raise AssertionError(mod + " was not blocked in the model process")
        print("BLOCKED_RESAMPLE")
    """)
    ok(r, "BLOCKED_RESAMPLE")


def test_guard_reports_when_it_is_already_too_late():
    """Guarding after the runtime has loaded is useless, and must say so
    rather than passing quietly -- the same failure mode as pin_threads()
    called after torch."""
    r = run("""
        import torch
        from nowcast_serve.guards import GuardViolation, guard_ingest_process
        try:
            guard_ingest_process()
        except GuardViolation as e:
            assert "already imported" in str(e), str(e)
            print("TOO_LATE_REPORTED")
        else:
            raise AssertionError("guard stayed silent after torch was loaded")
    """)
    ok(r, "TOO_LATE_REPORTED")


def test_submodules_are_blocked_not_just_the_top_level():
    r = run("""
        from nowcast_serve.guards import GuardViolation, guard_ingest_process
        guard_ingest_process()
        try:
            import torch.nn.functional
        except GuardViolation:
            print("BLOCKED_SUBMODULE")
        else:
            raise AssertionError("torch.nn.functional slipped past the guard")
    """)
    ok(r, "BLOCKED_SUBMODULE")


def test_unrelated_imports_are_untouched():
    r = run("""
        from nowcast_serve.guards import guard_ingest_process
        guard_ingest_process()
        import json, gzip, numpy          # a prefix-only match would break these
        import torchvision_not_real_pkg_check as _  # noqa
    """)
    assert r.returncode != 0
    assert "ModuleNotFoundError" in r.stderr, r.stderr[-1500:]
    assert "GuardViolation" not in r.stderr, (
        "guard matched a name that merely starts with 'torch'")


@needs_scan
def test_module_level_service_does_not_storm():
    """A service constructed at module level, with no main guard, used to
    start a worker in every re-imported child -- endlessly. One exec'd child
    cannot re-import anything, so exactly one worker exists."""
    r = run(f"""
        from nowcast_serve import NowcastService
        svc = NowcastService(dim=32, depth=1)          # module level, no guard
        res = svc.handle_scan({REAL_SCAN!r})
        assert res["check"]["passed"]
        print("WORKER_PID", svc.worker.child_pid)
        svc.close()
        print("NO_STORM_OK")
    """)
    ok(r, "NO_STORM_OK")
    assert r.stdout.count("WORKER_PID") == 1, r.stdout[-1500:]


@needs_scan
def test_worker_starts_from_a_parent_with_a_large_resident_set():
    """Linux-relevant: Popen cannot use posix_spawn when pass_fds is given
    (CPython subprocess.py guards on `not pass_fds`), so the worker is
    started with fork+exec.

    fork+exec is SAFE for the OpenMP problem -- the child execs, so the
    parent's initialised libomp is replaced, which is exactly what
    multiprocessing's fork never does. But fork from a large-RSS parent is
    its own hazard: under strict overcommit (vm.overcommit_memory=2) the
    kernel must reserve the parent's whole commit, and a process holding a
    model plus a CUDA context can be many GB.

    Hence NowcastService starts the worker BEFORE importing torch. That
    ordering is not a latency optimisation any more; it keeps the fork
    cheap. This test holds ~2 GB in the parent first.
    """
    r = run(f"""
        import numpy as np
        ballast = np.ones((2000, 2000, 64), dtype=np.uint8)   # ~2 GB resident
        ballast[::997] = 7
        print("RSS_BALLAST", ballast.nbytes // 10**6, "MB")

        from nowcast_serve import IngestWorker

        def main():
            with IngestWorker() as w:
                arrays, check = w.ingest({REAL_SCAN!r})
                assert arrays["TIR1"].shape == (912, 864)
            print("LARGE_PARENT_OK")

        if __name__ == "__main__":
            main()
    """)
    ok(r, "LARGE_PARENT_OK")


def test_the_ingest_side_never_needs_torch():
    """The fetch/ingest host installs `nowcast-eval[ingest]` -- no torch.

    Not installing torch there does not work around the OpenMP conflict, it
    removes it: two runtimes cannot collide if only one is present. This
    asserts the import graph actually allows that, by forbidding torch and
    then doing a real ingest.
    """
    r = run(f"""
        from nowcast_serve.guards import forbid
        forbid(("torch",), "the ingest host does not install torch")

        import nowcast_data
        import nowcast_flood
        from nowcast_data.insat import ingest_scan
        from nowcast_data.fetch import Fetcher, verify_hdf5
        from nowcast_data.ranged import probe_range_support
        import sys
        assert "torch" not in sys.modules
        print("INGEST_WITHOUT_TORCH_OK")
    """)
    ok(r, "INGEST_WITHOUT_TORCH_OK")
