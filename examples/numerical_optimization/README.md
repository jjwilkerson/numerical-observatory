# Optional numerical-metric acceleration

This phase optimizes GMDS evaluation, not the metric model or geodesic solver.
Read [PERFORMANCE.md](PERFORMANCE.md) for controlled measurements and accuracy
results. The [Phase 1 audit](../numerical_spacetime/ARCHITECTURE.md) remains the
algorithm specification; its coverage certification and data loader are unchanged.

## Backend selection

Add one field to the existing numerical-spacetime configuration:

```toml
[spacetime]
type = "numerical"
dataset = "../../reference/geodesics-cpp/results/amr-composite-multilevel/carpetx-native-gmds"
spatial_interpolation = "cubic_amr_c1_hierarchy"
backend = "numba"
```

| Backend | Implementation |
|---|---|
| `python` | Original evaluator and reconstruction; unchanged default/reference |
| `numpy` | Cached patch geometry, one-patch gather optimization, broadcast 3+1 algebra; original seam/overlap path |
| `numba` | Compiled float64 spatial and temporal ADM interpolation; NumPy/LAPACK 3+1 algebra with broadcast derivatives |

All five spatial modes are supported. Backend selection changes neither camera
initialization nor solver/tolerances. Numerical data still requires Hamiltonian
integration; Carter, Schwarzschild quadrature, symmetry reuse and disks remain
unsupported. Select the backend at construction time; create a new provider to
switch it. Python API:

```python
from numerical_observatory.numerical_metric import NumericalMetric
metric = NumericalMetric('output/my-gmds', 'cubic_amr_c1_hierarchy', backend='numba')
inverse, derivatives = metric.evaluate([4.0, 6.5, 0.23, 0.17])
```

Install the optional CPU-only dependency, Windows PowerShell:

```powershell
.\.venv\Scripts\python -m pip install -e '.[test,acceleration]'
.\.venv\Scripts\python -m numerical_observatory --config examples/numerical_optimization/real_preview_numba.toml --workers 2 --output output/numerical-numba-preview
```

Ubuntu:

```bash
.venv/bin/python -m pip install -e '.[test,acceleration]'
.venv/bin/python -m numerical_observatory --config examples/numerical_optimization/real_preview_numba.toml --workers 2 --output output/numerical-numba-preview
```

The `acceleration` extra installs Numba 0.67 and compatible llvmlite. It is
independent of the optional `carpetx` native-I/O extra. Base installation and all
analytical spacetimes need neither Numba nor LLVM/CUDA. No compiler/build system
is added. See the [official Numba release notes](https://numba.readthedocs.io/en/0.67.0/release/0.67.0-notes.html)
for its NumPy 2.5 support. The measured environment uses Python 3.13.7 and
NumPy 2.5.4; other platforms need their normal regression run.

## Numerical and failure semantics

Compiled kernels use float64, `fastmath=False`, NumPy division semantics and no
parallel reductions. They retain the centered cubic weights and analytic
derivatives, explicit plane/box branches, required-level pruning, interpolation
before metric inversion, and derivatives of blend weights. A fixed hierarchy is
certified by the original constructor before any kernel runs.

Native lapse/SPD/finite validation still runs globally on each active snapshot
pair. Per-query independent interpolants, composites and temporal fields are
checked. A conservative near-singular Cholesky guard or a kernel rejection
delegates the **whole query** to the original evaluator. That evaluator decides
whether it is valid and supplies the original error code. Kernel attempts are
not double-counted as successful stencils. This is an explicit fallback, not
coarser extrapolation or clipping. Overlap checks must complete before deciding
that a stencil is missing; linear fallback cannot hide inconsistent samples.

`evaluation_backend` and `acceleration_policy` appear in metric metadata.
Per-ray `metric_backend_fallbacks` and the aggregate count record reference
consultations, including rejected queries. Missing Numba raises an informative
error when the backend is used; it never silently selects another backend.

The compiled 3+1 reconstruction remains a **benchmark-only prototype**. Small
LAPACK/reduction differences changed adaptive histories in strong-field tests.
Production `numba` therefore keeps NumPy/LAPACK reconstruction, reusing the
validated broadcast algebra of the `numpy` backend. See the
report for the failed prototype checks and final strong-ray comparisons.
No production integration tolerance, temporal guard, null criterion, ray status,
or photon momentum was adjusted for speed or to repair a trajectory.

## Initialization, memory and workers

The first query can include Numba import, typed-list initialization and JIT
compilation/cache loading. It is not steady-state throughput. Subsequent worker
processes independently initialize their runtime and load cached machine code.
The source cache is managed by Numba; set `NUMBA_CACHE_DIR` to a writable local
directory if an installed package is read-only. Cache binaries are machine and
software dependent and are not scientific ray data.

Kernels read views of the existing read-only memory maps. No packed copy of the
dataset is created. Only the active snapshot pair is retained, and kernel views
are removed from the provider's pickle state. Windows `spawn` reopens maps in
each worker. OS file-backed pages may be shared, but Numba/LLVM runtime/code and
validation scratch add per-process memory. Do not sum worker RSS and call it
unique physical memory. Fewer workers can be faster on small images.

Scratch arrays are local to evaluations. Providers retain the original mutable
single-event memo/statistics, so use independent providers for concurrent
threads; the existing multiprocessing implementation already does this.
Optional paths use an equivalent binary search for time-slab indices. Compiled
views are reused only while the active snapshot-pair identities match; changing
or externally reactivating maps invalidates the views. Field values, time weights
and derivatives are always calculated at the requested event.

## Reproducing benchmarks

Run benchmarks without a simultaneous test suite or other heavy work. The suite
below launches backends sequentially in separate interpreter processes:

```powershell
.\.venv\Scripts\python -m numerical_observatory.numerical_benchmark --suite --output output/numerical-bench --real reference/geodesics-cpp/results/amr-composite-multilevel/carpetx-native-gmds --count 512 --repeats 3
```

On Ubuntu replace `.\.venv\Scripts\python` with `.venv/bin/python`.
The suite evaluates all five modes on manufactured data, actual three-level
CarpetX data, split-patch seams, five geodesic controls, and 16-ray renders with
serial/requested-two-worker execution. Production batching uses 64 rays, so this
tiny render actually launches one child. The execution records expose this;
those render timings measure spawn overhead, not two-core scaling.
It deliberately does not launch large renders.

Individual commands:

```powershell
.\.venv\Scripts\python -m numerical_observatory.numerical_benchmark --backend python --output output/bench-python --seams --count 1024 --repeats 5
.\.venv\Scripts\python -m numerical_observatory.numerical_benchmark --backend numba --output output/bench-numba --seams --count 1024 --repeats 5
.\.venv\Scripts\python -m numerical_observatory.numerical_benchmark --backend numba --output output/bench-real --real reference/geodesics-cpp/results/amr-composite-multilevel/carpetx-native-gmds --modes cubic_amr_c1_hierarchy --trajectories --renders --count 512 --repeats 3
# Actual two-process tracing, with benchmark-only four-ray batches:
.\.venv\Scripts\python -m numerical_observatory.numerical_parallel_benchmark --backend numba --dataset reference/geodesics-cpp/results/amr-composite-multilevel/carpetx-native-gmds --output output/parallel-numba --repeats 3
.\.venv\Scripts\python -m numerical_observatory.numerical_initialization_benchmark --dataset reference/geodesics-cpp/results/amr-composite-multilevel/carpetx-native-gmds --output output/native-initialization.json --repeats 5
```

Every JSON records deterministic query hashes, source hashes, dataset identity,
software/CPU/topology information, all repeat times, medians and variability.
Warm queries vary positions and times and include exact/adjacent-float interfaces
and temporal endpoints. Invalid queries are timed separately. Interior temporal
knots and rejected/time-domain cases are also covered by accuracy tests.
The parallel-batch command reuses the existing scheduler, verifies that two
processes actually launched, and requires exactly equal per-ray results and
diagnostics. Run it with `python` and `numpy` too for comparisons. Production
batching and multiprocessing behavior are unchanged.

Microbenchmarks separate spatial interpolation, reference stencil-coordinate
construction, reference/compiled gathering, field validation, reconstruction and
full Hamiltonian RHS. The compiled-reconstruction timing is explicitly a
prototype, not the selected production reconstruction. Temporary Python
allocations, logical mapped bytes, serialized provider size, process RSS and
lifetime peak RSS are distinct measurements. `tracemalloc` does not capture all
Numba/BLAS/native allocations. Peak RSS later in a process includes previous
cases and profiling; the first-query snapshot separates initialization cost.
The gather/stencil microbenchmarks use a common 64-node stencil for comparison
even in the linear case; complete linear interpolation timings use its actual
eight-node stencil. Component times overlap and should not be summed.
The initialization command separately times metadata, certification, lazy mmap
opening and the unchanged native-validation loop on warm file-backed pages.

For a genuinely cold JIT measurement use a new, empty cache directory:

```powershell
$env:NUMBA_CACHE_DIR = "$PWD/output/new-numba-cache"
.\.venv\Scripts\python -m numerical_observatory.numerical_benchmark --backend numba --output output/bench-cold --modes cubic_amr_c1_hierarchy --count 128 --repeats 3
Remove-Item Env:NUMBA_CACHE_DIR
```

Ubuntu: prefix the Python command with
`NUMBA_CACHE_DIR="$PWD/output/new-numba-cache"`. Do not delete another job's cache.

## Reproducing accuracy checks

```powershell
.\.venv\Scripts\python -m pytest -q
.\.venv\Scripts\python -m numerical_observatory.numerical_comparison --output output/numba-accuracy --backend numba --events --strong --count 1024 --real reference/geodesics-cpp/results/amr-composite-multilevel/carpetx-native-gmds --tolerances 1e-11 1e-12
.\.venv\Scripts\python -m numerical_observatory.numerical_validation --output output/numba-cpp --backend numba --oracle examples/numerical_spacetime/oracle_windows.json
```

The C++ oracle is optional and uses the existing validation bridge; see the
Phase 1 guide for compilation and Ubuntu oracle paths. Its source is untouched.
Saved C++ golden regressions run without the executable. Optional real/analytical
datasets are skipped explicitly by tests when absent. Base tests remain usable
without Numba; compiled tests skip that optional backend in such installations.

Rebuild the tables from the archived study (including its saved test log):

```powershell
.\.venv\Scripts\python tools/numerical_performance_report.py --data examples/numerical_optimization --tests examples/numerical_optimization/test_results.txt
```
