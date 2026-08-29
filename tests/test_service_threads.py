"""Single-process OpenMP pinning: training, cache builds, notebooks.

SCOPE NOTE. These tests cover `nowcast_data._threads`, which ORDERS around
the OpenMP conflict. The inference service no longer works that way -- it
runs ingest in an exec'd child that has never imported torch, and is covered
by tests/test_service_shape.py. What remains here is the single-process case
that legitimately does both jobs.

The original framing, still true for those callers:

The inference service does pyresample THEN torch in one process.

pykdtree (under pyresample) and torch each ship an OpenMP runtime, and two in
one process abort the interpreter. Pinning it in conftest.py fixes the test
suite and does nothing for deployment, so this test deliberately runs in a
SUBPROCESS with a clean environment -- conftest's fixtures cannot mask a
failure here, which is the whole point.
"""
import os
import subprocess
import sys
import textwrap

import pytest

# Exactly what an inference tick does: put a swath on the analysis grid, then
# run the model on it, in one process.
SERVICE_PATH = textwrap.dedent("""
    import nowcast_data                      # pins OpenMP before anything else
    import numpy as np
    from pyresample.geometry import SwathDefinition
    from pyresample.kd_tree import resample_nearest
    from nowcast_data.grids import india_area

    rng = np.random.default_rng(0)
    n = 600
    lat = np.linspace(5, 40, n)[:, None] * np.ones((1, n))
    lon = np.linspace(62, 102, n)[None, :] * np.ones((n, 1))
    bt = rng.normal(280, 15, (n, n))
    grid = resample_nearest(SwathDefinition(lons=lon, lats=lat), bt,
                            india_area(), radius_of_influence=20000,
                            fill_value=np.nan)
    assert grid.shape == (912, 864), grid.shape

    import torch
    from nowcast_model import ModelConfig, MultiTaskNowcaster
    m = MultiTaskNowcaster(ModelConfig(in_channels=2, context_frames=2,
            grid_size=96, lead_steps=6, dim=16, depth=1)).eval()
    x = torch.randn(1, 2, 2, 96, 96)
    co = torch.rand(1, 8, 2) * 2 - 1
    with torch.no_grad():
        out = m(x, co)
    assert set(out) == {"rain_rate", "extreme_rain", "cloudburst"}

    # and again, to be sure the abort is not merely deferred
    resample_nearest(SwathDefinition(lons=lon, lats=lat), bt, india_area(),
                     radius_of_influence=20000, fill_value=np.nan)
    print("SERVICE_PATH_OK")
""")


def _run(script: str, env_extra: dict | None = None) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items()
           if k not in ("OMP_NUM_THREADS", "PYKDTREE_NUM_THREADS",
                        "KMP_DUPLICATE_LIB_OK", "OPENBLAS_NUM_THREADS",
                        "MKL_NUM_THREADS")}
    env.update(env_extra or {})
    return subprocess.run([sys.executable, "-c", script], capture_output=True,
                          text=True, timeout=600, env=env,
                          cwd=os.path.dirname(os.path.dirname(__file__)))


def test_resample_then_infer_survives_in_one_process():
    """The production path. If this aborts, the service aborts."""
    r = _run(SERVICE_PATH)
    assert r.returncode == 0, (
        f"service path aborted (rc={r.returncode}).\n"
        f"stdout:\n{r.stdout[-2000:]}\nstderr:\n{r.stderr[-3000:]}")
    assert "SERVICE_PATH_OK" in r.stdout


def test_infer_then_resample_also_survives():
    """Reverse order: torch's OpenMP loads first. Importing nowcast_data after
    torch is too late for the variables, so this is the case the entrypoint
    guidance exists for -- it must still not abort."""
    reversed_path = ("import torch\n" + SERVICE_PATH)
    r = _run(reversed_path)
    assert r.returncode == 0, (
        f"reverse order aborted (rc={r.returncode}).\n{r.stderr[-3000:]}")


def test_pinning_is_applied_by_importing_the_package():
    r = _run("import nowcast_data, os; "
             "print(os.environ.get('OMP_NUM_THREADS'), "
             "os.environ.get('PYKDTREE_NUM_THREADS'))")
    assert r.returncode == 0, r.stderr[-2000:]
    assert r.stdout.split() == ["1", "1"]


def test_operator_override_is_respected():
    """An operator who sets OMP_NUM_THREADS deliberately must not be
    silently overridden."""
    r = _run("import nowcast_data, os; print(os.environ['OMP_NUM_THREADS'])",
             env_extra={"OMP_NUM_THREADS": "4"})
    assert r.returncode == 0, r.stderr[-2000:]
    assert r.stdout.strip() == "4"


def test_opt_out_is_honoured():
    r = _run("import nowcast_data, os; "
             "print(repr(os.environ.get('PYKDTREE_NUM_THREADS')))",
             env_extra={"NOWCAST_NO_THREAD_PIN": "1"})
    assert r.returncode == 0, r.stderr[-2000:]
    assert r.stdout.strip() == "None"


# --------------------------------------------------------------------------
# The real ingest path, not a synthetic swath
# --------------------------------------------------------------------------

REAL_SCAN = ("/Users/evad/test_ff/3RIMG_L1B_STD/2025/01AUG/"
             "3RIMG_01AUG2025_2345_L1B_STD_V01R00.h5")

REAL_INGEST_PATH = textwrap.dedent(f"""
    from nowcast_data._threads import ensure_pinned_or_reexec
    ensure_pinned_or_reexec()
    import nowcast_data
    from nowcast_data.insat import ingest_scan
    arrays, chk = ingest_scan({REAL_SCAN!r}, ("TIR1", "WV"), strict=False)
    assert chk.passed
    import torch
    _ = torch.backends.mps.is_available()      # GPU probe AFTER resampling
    from nowcast_model import ModelConfig, MultiTaskNowcaster
    m = MultiTaskNowcaster(ModelConfig(in_channels=2, context_frames=2,
            grid_size=96, lead_steps=6, dim=16, depth=1)).eval()
    with torch.no_grad():
        m(torch.randn(1, 2, 2, 96, 96), torch.rand(1, 4, 2) * 2 - 1)
    print("REAL_PATH_OK")
""")


@pytest.mark.skipif(not os.path.exists(REAL_SCAN), reason="real 3DR scan absent")
def test_real_ingest_then_infer_survives():
    """The gap this closes: the synthetic test above uses
    pyresample.kd_tree.resample_nearest directly, but ingest_scan goes through
    SATPY's resample path. The synthetic test passed while the real service
    path aborted with 'OMP: Error #15' -- a test that was green while
    production died.

    It also pins the ordering finding: probing the GPU
    (torch.backends.mps.is_available) before the resampling is what triggers
    it, so ingest must complete first.
    """
    r = _run(REAL_INGEST_PATH)
    assert r.returncode == 0, (
        f"real ingest+infer path aborted (rc={r.returncode})\n"
        f"stdout:\n{r.stdout[-1500:]}\nstderr:\n{r.stderr[-2500:]}")
    assert "REAL_PATH_OK" in r.stdout


def test_reexec_sets_the_variables_even_when_stripped():
    """os.environ from Python is too late for some libraries; the re-exec is
    the only reliable form."""
    r = _run("from nowcast_data._threads import ensure_pinned_or_reexec\n"
             "ensure_pinned_or_reexec()\n"
             "import os\n"
             "print(os.environ.get('KMP_DUPLICATE_LIB_OK'), "
             "os.environ.get('_NOWCAST_THREADS_PINNED'))")
    assert r.returncode == 0, r.stderr[-1500:]
    assert r.stdout.split() == ["TRUE", "1"]


def test_reexec_does_not_loop():
    r = _run("from nowcast_data._threads import ensure_pinned_or_reexec\n"
             "ensure_pinned_or_reexec(); ensure_pinned_or_reexec()\n"
             "print('no loop')",
             env_extra={"_NOWCAST_THREADS_PINNED": "1"})
    assert r.returncode == 0 and "no loop" in r.stdout
