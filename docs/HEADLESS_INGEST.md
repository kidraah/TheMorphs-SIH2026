# Running the archive pull headless on a Linux VPS

## The short version

The fetch/ingest host **does not need torch**, and that is the main thing to
get right. Not installing it does not work around the pykdtree/torch OpenMP
conflict — it removes it, because two runtimes cannot collide when only one
is present. `nowcast_data`, `nowcast_flood` and the fetcher import no torch
at module level (asserted by a test that forbids the import and then runs a
real ingest).

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[ingest,era5,flood]"        # NOT [model]
```

`torch` and `matplotlib` are now optional extras rather than base
dependencies, for exactly this reason.

## Does the exec'd subprocess boundary behave the same on Linux?

**Yes, and for a reason that survives the platform difference.**

The hazard flagged earlier is real but specific: CPython picks the
`multiprocessing` start method by platform — `spawn` on darwin, **`fork`
everywhere else** (`Lib/multiprocessing/context.py`) — and a forked child
inherits the parent's initialised libomp and SIGABRTs. Measured: exit code
−6.

`IngestWorker` does not use `multiprocessing` at all. It runs

```python
subprocess.Popen([sys.executable, "-m", "nowcast_serve.ingest_worker",
                  "--fd", str(fd)], pass_fds=(fd,))
```

which **execs**. Whether the kernel gets there via `fork+exec` or
`posix_spawn` is irrelevant to OpenMP: the child's address space is replaced
by a fresh image, so nothing is inherited. That is precisely what
`multiprocessing`'s fork never does. A test asserts `Process(` does not
appear in `pool.py`, so this cannot regress silently.

**One Linux-specific consequence worth knowing.** Reading CPython's
`subprocess.py`, the `posix_spawn` fast path is guarded on `not pass_fds`
among other conditions — and we pass fds. So on Linux this is `fork+exec`,
not `posix_spawn`. `fork` from a large-RSS parent is cheap under the default
`vm.overcommit_memory=0` (copy-on-write) but can fail with ENOMEM under
strict overcommit (`=2`), because the kernel must reserve the parent's whole
commit and a process holding a model plus a CUDA context is many GB.

Mitigation is already in the code and is now commented as load-bearing:
`NowcastService` starts the ingest worker **before** importing torch, so the
fork happens while the parent is small. A test starts the worker from a
parent holding ~2 GB.

On the fetch host none of this applies, because torch is not installed.

## Dependencies

| host | install | why |
|---|---|---|
| fetch / ingest VPS | `.[ingest,era5,flood]` | no torch, no matplotlib |
| training box | `.[model,ingest,era5,flood]` | needs both |
| dev | `.[dev]` | pulls everything |

System packages the wheels need on a bare Debian/Ubuntu box: `libhdf5-dev`,
`libgeos-dev`, `libproj-dev`, `build-essential`. `rasterio` and `h5py` ship
manylinux wheels, so usually none of these are required — install them only
if pip falls back to building from source.

## Thread pinning still applies on the fetch host

The service split removes the constraint for the *inference* process. The
fetch host runs decode in the same process as everything else, so
`nowcast_data._threads` still governs it. It is applied on package import,
and `ensure_pinned_or_reexec()` is the reliable form for an entrypoint:

```bash
export OMP_NUM_THREADS=1 PYKDTREE_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE
```

Setting them in the shell (or the systemd unit) is strictly better than
relying on the in-process re-exec, because OpenMP reads them when its
runtime loads. With torch absent the conflict cannot arise, but
single-threaded resampling is the right default regardless: the work is one
KD-tree query per scan and parallelism belongs to the pull, not the resample.

## Disk sizing

**Raw never accumulates.** 3.4 TB passes through a few GB of staging, because
each scan is fetched to `.part`, verified, decoded to the 4 km grid, and the
raw file deleted.

| | cache | staging | recommend |
|---|---|---|---|
| Range honoured | 117 GB | ~1 GB | **150 GB** |
| Range not honoured | 117 GB | ~4 GB (8 parallel × 500 MB) | **160 GB** |

Cache figure is `InsatCacheConfig.archive_gb(8000)` — 15.8 MB/scan for six
channels plus the antecedent channel and finite masks, uint16.

Add whatever you want for IMERG and the static layers; those are separate
and much smaller. **A 250 GB volume is comfortable; 500 GB is generous.**
Do not size for 3.4 TB — if you find yourself needing to, the
decode-and-discard step has stopped running and raw is piling up. Worth an
explicit disk-usage alarm for that reason.

## Running it unattended

* **systemd, not `nohup`.** `Restart=on-failure`, `RestartSec=60`. The
  fetcher is idempotent: on restart it skips complete files, re-fetches any
  whose file is missing, and resumes partials where the server allows.
* **`RuntimeMaxSec`** is not wanted — a 5.3-day run is expected.
* **Watch the manifest, not the log.** `Manifest.report()` gives complete /
  pending / failed, GB written, mean MB/s, and the reason for each failure.
* **Sweep orphaned `.part` files** with `sweep_orphaned_parts()` before each
  restart. The event pull left seven; they are harmless but they mean a
  crash happened and are worth counting.
* MOSDAC's token expires — the event pull ended with
  `Invalid Refresh Token`. Over 5.3 days that will happen repeatedly, so the
  fetcher must be able to re-authenticate mid-run, not only at start. That
  is the one piece **not yet built**, because the refresh handshake has not
  been seen from this side; it should be lifted from the working downloader
  rather than guessed at.

## Measured numbers this planning rests on

| | value | source |
|---|---|---|
| per-scan delivered | 448 MB | 702 files, 292 GB |
| sustained rate | **7.9 MB/s** | 292 GB in 10.5 h |
| failure rate | 1 × HTTP 500, 7 orphaned `.part` | one 10.5 h run |
| ranged saving | 25.5× byte-identical | real scan, local server |
| archive, no Range | 3,596 GB / 135.8 h | measured rate |
| archive, Range | 141 GB / 5.3 h | measured rate |
