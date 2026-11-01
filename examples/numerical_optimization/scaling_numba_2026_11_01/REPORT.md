# Numba numerical-spacetime multiprocessing scaling

Measurement date: 2026-11-01. Production algorithms, interpolation, scheduler, batch size, integration tolerances and C++ reference sources were not edited. Production Python source hashes are identical before/after every run and across all configurations.

Hardware/software: Intel(R) Core(TM) i9-10850K CPU @ 3.60GHz, 10 physical / 20 allowed logical CPUs, Windows 11, Python 3.13.7, NumPy 2.5.4, SciPy 1.18.1 and Numba 0.67.0. The same existing Numba disk cache was used throughout.

## Scene and method

Real three-level CarpetX GMDS dataset; `cubic_amr_c1_hierarchy`, `backend="numba"`, static observer at t=4, (6.5,0.23,0.17), looking along +X. Horizontal FOV 40°, 40×32 pixels, one ray per pixel, celestial-grid background, explicit finite direction sphere R=8, no capture surface. rtol=1e-10, atol=1e-12, max step=0.06, affine limit=6. The numerical observation is finite-boundary, not an asymptotic black-hole shadow.

The 32×24 pilot used five workers and took 138.35 s; its 12 batches were insufficient for multiple batches at ten workers. The final 40×32 grid has 1,280 rays and 20 unchanged 64-ray production batches. A final-scene ten-worker warm-up took 123.01 s; every worker completed exactly two batches. All kernels used the existing disk cache. The instrumented 4×4 prewarm exactly matched the prior saved production arrays.

Two measured repetitions ran sequentially in orders **1→2→5→10**, then **10→5→2→1**. No CPU-heavy Observatory tests or simulations ran concurrently. CPU frequency/temperature and unrelated system activity were not controlled. A fresh interpreter and fresh production worker pool were used for each run. No batch policy was altered.

The primary wall time is measured around the entire child command: interpreter/parent setup, cached-kernel preparation, worker startup, tracing, assembly, output, hashes and record writing. Render-only wall time and all other phase timings are retained separately. One worker means the normal serial parent (zero child processes); larger counts are verified active child workers.

## Scaling

| Workers | Actual workers | Median wall time | Speedup vs. 1 | Parallel efficiency | Peak memory |
|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 729.01 s | 1.00× | 100.0% | ≤169 MiB summed RSS peaks; 169 MiB/compute process |
| 2 | 2 | 371.28 s | 1.96× | 98.2% | ≤495 MiB summed RSS peaks; 166 MiB/compute process |
| 5 | 5 | 176.65 s | 4.13× | 82.5% | ≤991 MiB summed RSS peaks; 166 MiB/compute process |
| 10 | 10 | 120.75 s | 6.04× | 60.4% | ≤1798 MiB summed RSS peaks; 164 MiB/compute process |

**Recommendation for this tested scene: 10 workers**, based on the lowest measured median wall time. This recommendation does not establish an optimum for larger datasets, different ray costs, more RAM, or different task granularity. Two repetitions describe observed variability, not a statistical confidence interval.

Memory is a conservative sum of each process’s individual peak RSS, not a measured simultaneous unique-memory peak. Shared libraries and file-backed mappings can be counted repeatedly. Native payloads remain read-only memory maps and are not serialized as full arrays.

## Individual timings and initialization

| Workers | Total wall repetitions, s | Median render wall, s | Parent provider + cached-kernel setup, s | Pool-ready span, s | First metric lookup per child, s |
|---:|---:|---:|---:|---:|---:|
| 1 | 733.56, 724.47 | 725.51 | 1.73 | 0.00 | parent only |
| 2 | 375.07, 367.49 | 367.64 | 1.83 | 3.38 | 0.91–1.01 |
| 5 | 177.14, 176.15 | 172.97 | 1.82 | 9.38 | 0.94–1.16 |
| 10 | 120.00, 121.50 | 117.22 | 1.74 | 20.02 | 0.86–1.37 |

Pool-ready span measures spawn/unpickling/core imports before the existing initializer announces readiness. Lazy Numba runtime/cache loading, typed-list construction and the initial native-snapshot validation occur at first metric use inside the first task. Those times are measured once per process by a pass-through observer; they overlap between workers and must not be summed/subtracted as a global critical-path time. Later snapshot mapping/validation remains part of real ray-tracing work. No root-kernel cache misses were observed in measured runs.

## Work, CPU use and numerical equivalence

| Workers | Completed scheduled batches | Per-child batch range | Approx. aggregate CPU time, s | Average busy logical CPUs | Task wall sum less first metric, s |
|---:|---:|---:|---:|---:|---:|
| 1 | 0 | serial, no scheduler tasks | 726.17 | 1.00 | not separately observable |
| 2 | 20 | 10–10 | 728.58 | 1.96 | 723.02 |
| 5 | 20 | 4–4 | 822.45 | 4.66 | 808.61 |
| 10 | 20 | 2–2 | 989.91 | 8.20 | 963.60 |

The serial renderer does not use scheduled batches; its 0 actual batches correspond to the same 20 batch-equivalents of ray work. Aggregate CPU time approximates parent lifetime CPU plus the final sampled lifetime CPU of each child; small shutdown/record-writing tails are excluded. Busy logical CPUs = aggregate CPU / total wall time (1.0 = one fully busy logical CPU), not physical-core utilization. Worker task-wall sums are aggregate work across concurrent processes and may exceed wall time.

Every measured run traced **1,280 rays** with **1,449,119 RHS evaluations**. Classifications: `{'captured': 0, 'escaped': 1280, 'unresolved': 0}`. Termination reasons: `{'numerical_finite_background': 1280}`. Backend fallback count: **0**. All saved per-ray fields, null residuals, statuses and classifications were exactly equal across all eight runs, as were rendered PNGs. Per-array hashes include dtype, shape and contiguous data bytes; full NPZ/PNG/diagnostic file hashes are also retained. Execution-specific diagnostic hashes may differ because timing/process provenance is included. No numerical tolerances were relaxed and no failures were omitted.

## Interpretation and limits

Two, five and ten workers completed exactly ten, four and two batches per child respectively, so there is no gross worker underuse or batch-count imbalance. Equal batch counts do not imply equal costs. Five→ten workers reduced median wall time by about 31.6%, rather than an ideal 50%. The pool-ready span grew from about 9.4 to 20.0 s, and aggregate CPU time increased from about 822 to 990 s for the same work. Initialization and increased per-work CPU cost are therefore visible costs; this is not evidence of a changed numerical workload.

Use the measured wall time, CPU-equivalent use, task distributions and phase costs together. Initialization cannot be assumed negligible merely because kernels are cached: each spawned process loads its own runtime and validates its first active snapshot pair. Two 64-ray tasks per process at ten workers limit dynamic load balancing; the final completion tail reflects the slowest task wave. Shared memory bandwidth, reduced all-core clock rates, OS activity and runtime overhead can reduce efficiency; these data do not independently identify a unique hardware bottleneck. There is no evidence here that Hyper-Threading doubles capacity, and 20 workers were not tested.

## Audit and reproduction

The measurement harness wraps only the first metric lookup (then restores the original method) and completed production tasks. It does not alter results, task sizes, scheduling bounds, worker initialization, inputs, tolerances or algorithms. One small JSON event is appended per task outside the original task timer; total wall includes that observational I/O. All production-source hashes match before/after each run. The C++ reference and full regression suite were not modified or rerun.

Windows PowerShell, from the repository root:
```powershell
.\.venv\Scripts\python examples/numerical_optimization/scaling_numba_2026_11_01/benchmark_scaling.py --study --config examples/numerical_optimization/scaling_numba_2026_11_01/scene.toml --output output/numba-scaling-repeat
.\.venv\Scripts\python examples/numerical_optimization/scaling_numba_2026_11_01/report_scaling.py
```
On Ubuntu substitute `.venv/bin/python`. Use a fresh output directory; prior measurements are never overwritten. Run the harness first with `--launch --config examples/numerical_optimization/scaling_numba_2026_11_01/scene.toml --workers 10 --output output/numba-scaling-warmup` before a new study. The existing Numba cache must be available; the study rejects measured root-kernel cache misses.

`scene.toml`, pilot/prewarm configurations, raw worker JSONL observations, all render products, per-run records, output hashes and rotated order are archived here. `measured/study.json` is the complete machine-readable measurement manifest. No Git commands or CarpetX simulations were run.
