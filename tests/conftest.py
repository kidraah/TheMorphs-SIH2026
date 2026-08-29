"""Test-wide configuration.

Two threading hazards, both of which abort the interpreter when the whole
suite runs in one process while each test passes in isolation:

1. pykdtree (under pyresample's nearest-neighbour resampling) and torch each
   bring their own OpenMP runtime. Two OpenMP runtimes in one process is a
   known abort. These variables must be set BEFORE either library loads, and
   conftest is imported before the test modules, so this is the right place.

2. dask's threaded scheduler, pulled in by satpy/xarray. The synchronous
   scheduler avoids the pool entirely and is the better default for tests
   anyway: deterministic, and these arrays are small.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("PYKDTREE_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import dask  # noqa: E402
import pytest  # noqa: E402


@pytest.fixture(autouse=True, scope="session")
def _dask_synchronous():
    with dask.config.set(scheduler="synchronous"):
        yield
