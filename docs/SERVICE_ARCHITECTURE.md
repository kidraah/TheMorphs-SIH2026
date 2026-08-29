# Service architecture: why ingest runs in its own process

## The failure

pykdtree (under pyresample, and so under satpy) and torch each ship their own
OpenMP runtime. Two initialised in one process abort the interpreter:

```
OMP: Error #15: Initializing libomp.dylib, but found libomp.dylib already initialized.
```

Not an exception — SIGABRT, exit code −6. It cannot be caught where it
happens, it does not appear in a traceback, and it takes the process with it.

The inference service does exactly the thing that triggers it: resample an
INSAT scan onto the analysis grid, then run a torch model on it.

## The workaround we had, and why it wasn't enough

`nowcast_data._threads` **ordered around** the conflict: pin the OpenMP
variables (re-exec'ing the interpreter if they were missing at startup), and
finish all CPU-side ingest before touching the GPU.

It worked. It also had a standing cost that made it the wrong answer:

- **The ordering had to be preserved by everything downstream.** Gunicorn, a
  container entrypoint, a scheduler, a future refactor — each had to keep the
  re-exec intact and each had to know that CPU ingest comes first.
- **It forbade how services are normally written.** Build the model at boot,
  log the device, then handle requests. That ordinary shape — GPU before the
  first scan — was the aborting one.
- **Nothing enforced it.** The rule lived in a docstring.

## The fix: separate processes

The two runtimes never meet, because they are never in the same process.

```
  model process                         ingest process
  ─────────────                         ──────────────
  torch, CUDA/MPS, the model            satpy, pyresample, pykdtree, h5py
  guarded against pyresample            guarded against torch
                    │                        │
                    └──── socketpair ────────┘
                       {"op": "ingest", ...}
                       {"ok": true, "arrays": {...}, "check": {...}}
```

`nowcast_serve.NowcastService` builds the model at boot, on the GPU, first
thing — the shape that used to abort — and delegates every decode and regrid
to a persistent child.

ERA5's regrid is pyresample too, so it is on the child's side of the
boundary as well (`worker.era5(...)`). Putting only the satellite decode
behind the split would have left the conflict in place for every run that
uses cross-attention context.

## The child is an exec, not a multiprocessing.Process

This is the load-bearing detail, and the first version got it wrong.

`multiprocessing`'s **spawn** re-imports the parent's `__main__` in the child
to recover the target function. So module-level torch work in an entrypoint —
`import torch`, a device probe, a device log line, the ordinary first three
lines of a service — is **replayed inside the ingest worker**. The symptom
was the parent's startup line printing twice:

```
GPU_UP mps        <- the parent
GPU_UP mps        <- the ingest child
```

Both OpenMP runtimes were back in one process, and the split held only as
long as every future entrypoint kept its torch work behind
`if __name__ == "__main__":`. That is the same "someone has to remember"
property the split was built to remove. The same mechanism also meant a
service constructed at module level started a worker in every re-imported
child, endlessly.

So the child is launched as a **named module in a fresh interpreter**:

```python
subprocess.Popen([sys.executable, "-m", "nowcast_serve.ingest_worker",
                  "--fd", str(fd)], pass_fds=(fd,))
```

There is no `__main__` to replay, nothing to recurse into, and nothing
inherited from the parent's address space.

## Measurements

Worst case throughout: parent has torch up on MPS, **no thread pinning at
all** (`NOWCAST_NO_THREAD_PIN=1`, OpenMP variables stripped).

| boundary | result |
|---|---|
| none (one process, GPU first) | SIGABRT, `OMP: Error #15` |
| `fork` | child exitcode **−6**, same abort |
| `spawn` via `mp.Process` | works, but replays the parent's `__main__` |
| **exec'd module (this design)** | **works; nothing to replay** |

`fork` buys nothing because it inherits the parent's already-initialised
libomp. That matters more than it looks: CPython picks the multiprocessing
default **by platform** (`Lib/multiprocessing/context.py`) —

```python
if sys.platform == 'darwin':  _default_context = ... 'spawn'
else:                         _default_context = ... 'fork'
```

— so a `Process`-based design works on the development Mac and SIGABRTs on
the Linux box it deploys to. Same shape as every other bug this project has
hit: green where it is looked at, dead where it runs.

### Cost of the boundary

Same scan, median of 4 after warm-up:

| | seconds |
|---|---|
| in-process ingest | 2.282 |
| **cross-process ingest** | **2.191** |
| payload over the pipe | 6.30 MB/scan (2 × float32 912×864) |
| child start | 0.05 s, once at boot |

Within noise. Shared memory would buy lifetime bugs and no measurable time,
so the arrays go over the pipe.

## The guards

Defence in depth, and they earn it — the `__main__`-replay hole above
surfaced as a *named `GuardViolation`* rather than a SIGABRT, which is how it
was diagnosable at all.

- `guard_ingest_process()` — the child forbids `torch`.
- `guard_model_process()` — the parent forbids `pyresample`, `pykdtree`, `satpy`.

Both install a `sys.meta_path` finder that raises **before** the extension
module (and its OpenMP runtime) loads. Blocking afterwards would be useless:
the abort happens at load time. Guarding after the runtime is already
resident raises too, rather than passing quietly — the same failure mode as
`pin_threads()` called after torch.

## What `nowcast_data._threads` is still for

Training, `build_cache.py`, notebooks, tests — anything single-process that
legitimately does both jobs. `pin_threads()` and `ensure_pinned_or_reexec()`
remain correct there. **The service no longer depends on them.**

## Tests

`tests/test_service_shape.py` tests the **service shape**, not the code path.
That distinction is the point: `tests/test_service_threads.py` exercised
`pyresample.kd_tree.resample_nearest` directly while `ingest_scan` goes
through satpy's resample path, so the suite was green while the real service
aborted.

Every test there runs in a subprocess with the OpenMP variables stripped and
`NOWCAST_NO_THREAD_PIN=1`, so the test harness cannot mask a failure — if the
split is doing the work, they pass with no pinning at all. Entrypoints are
written to **files**, not `python -c`, because spawn's `__main__` recovery is
path-based and `-c` would exercise a different bootstrap than a deployed
entrypoint does.

Notable cases:

- `test_service_boots_model_then_handles_scans` — the shape that used to abort.
- `test_entrypoint_torch_at_module_level_is_not_replayed_in_the_worker` —
  asserts `GPU_UP` appears exactly once, from an entrypoint with no main guard.
- `test_fork_really_does_abort_so_the_exec_is_not_cargo_cult` — if it ever
  starts passing, the reasoning behind the design has changed.
- `test_service_survives_a_bad_scan_and_keeps_serving` — the split must not
  trade an abort for an outage.
