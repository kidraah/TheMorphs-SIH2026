"""The inference service does pyresample THEN torch in one process.

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
