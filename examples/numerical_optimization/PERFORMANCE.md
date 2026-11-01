# Numerical metric optimization: measured performance and validation

Measured 2026-11-01 on Intel(R) Core(TM) i9-10850K CPU @ 3.60GHz. Windows 11, Python 3.13.7, NumPy 2.5.4, SciPy 1.18.1, Numba 0.67.0 / llvmlite 0.49.0. The topology probe found 10 physical / 20 allowed logical CPUs. Python production tests ran on Windows; the independent C++ oracle ran under Ubuntu/WSL. A full Linux Python regression run was not performed.

## Result and implementation

The optional `numba` backend compiles all five spatial interpolation modes and temporal ADM interpolation, while retaining NumPy/LAPACK reconstruction with validated broadcast derivatives. `python` remains the unchanged default/reference path; `numpy` offers a dependency-free smaller gain. No integration tolerance, camera, physical field convention, native-data validation, coverage certificate or geodesic equation was relaxed. There is no GPU, disk or dynamic-AMR implementation in this phase.

Production changes are confined to optional metric dispatch, cached immutable geometry/views, strict CPU kernels, equivalent temporal-index binary search, vectorized algebra and additional provenance/counters. The data loader, AMR coverage-certification module and nine analytical/camera/disk modules retain their prior source hashes (`preserved_sources.json`). The DOP853 stepping code is unchanged; its additions only record metric backend/fallback counts.

## Controlled method

The authoritative `controlled-final/` runs used sequential isolated interpreter processes, 512 deterministic distinct/stratified events, three repeats, and identical dataset/query hashes and valid-query sets across backends. No test suite or other Observatory compute job was launched alongside these benchmarks; unrelated system activity was not controlled. All repeat times, variability, CPU/software/source hashes and rejection counts are retained in JSON. OS page caches were not evicted. Early `baseline/`, `numpy/` and `controlled/` runs document stages; the final table below is the matched final comparison.

## Warm metric evaluations

| Workload | Python eval/s | NumPy eval/s | NumPy speedup | Numba eval/s | Numba speedup |
|---|---:|---:|---:|---:|---:|
| synthetic linear | 1,721 | 2,023 | 1.18× | 7,332 | 4.26× |
| synthetic cubic | 1,591 | 2,067 | 1.30× | 6,193 | 3.89× |
| synthetic cubic_amr_c1 | 1,373 | 1,453 | 1.06× | 6,108 | 4.45× |
| synthetic cubic_amr_c1_box | 1,147 | 1,380 | 1.20× | 6,041 | 5.27× |
| synthetic cubic_amr_c1_hierarchy | 1,121 | 1,303 | 1.16× | 5,757 | 5.13× |
| real cubic_amr_c1_hierarchy | 1,160 | 1,349 | 1.16× | 6,017 | 5.19× |

The ≥5× target is met on the representative manufactured and real three-level hierarchy workloads; the ≥10× metric stretch target was not met. All five modes improve; the slowest relative compiled gain is still about 3.9×. Rejected requests are timed separately and consult the reference evaluator; their counts/reasons agree. These are complete metric-and-four-derivative calls, not repeated hits on the single-event memo.

| Real hierarchy repeat variability | Median loop seconds / 512 events | Standard deviation |
|---|---:|---:|
| python | 0.441519 | 0.007090 |
| numpy | 0.379619 | 0.007147 |
| numba | 0.085092 | 0.001445 |

## Component costs and profiling

| Real hierarchy component | Python µs/operation | Selected compiled-path µs/operation |
|---|---:|---:|
| Single-snapshot spatial interpolation | 311.75 | 28.67 |
| 64-node gather | 46.25 | 14.77 |
| Native NumPy final-interpolant validity check | 16.92 | 17.05 |
| 3+1 reconstruction | 134.32 | 90.27 |
| Full bare Hamiltonian RHS | 903.42 | 212.46 |

The gather/stencil-construction microbenchmarks use a common 64-node stencil even for linear mode; its complete interpolation timing uses the correct eight nodes. Spatial timings include the reference-required independent/composite validity checks. The final NumPy validity check is deliberately retained. Component timings overlap and must not be summed. The bare-RHS workload uses an arbitrary fixed covector to measure all algebra; integrated rays use physically null camera initial conditions.

Initial profiling identified interpolation, node gathering, repeated small allocations and validation as the dominant costs. The first low-risk NumPy stage gave a modest gain. A fully compiled reconstruction prototype was faster but failed strong-ray equivalence (see below), so production keeps NumPy/LAPACK algebra. A separate 512-event reconstruction experiment found the broadcast NumPy version bitwise equal to the original and 1.75× faster (`reconstruction_choice.json`). Final profiling then identified scalar time-index clipping and repeated view bookkeeping; equivalent binary search and map-identity caching raised real throughput from about 5,012 to 6,017/s. New tests force exact/one-ULP knot queries and external map reactivation to verify invalidation.

After optimization, NumPy reconstruction/final validation and Python/kernel dispatch are prominent remaining costs. Compiled spatial interpolation alone is about 10.9× faster on this workload; that is **not** the complete metric or renderer speedup.

## End-to-end geodesics

| Control | Python s | NumPy s | Numba s | Numba speedup |
|---|---:|---:|---:|---:|
| real_escape | 1.558 | 1.415 | 0.592 | 2.63× |
| schwarzschild_capture | 8.743 | 7.675 | 3.018 | 2.90× |
| schwarzschild_escape | 12.747 | 11.204 | 4.398 | 2.90× |
| kerr05_capture | 8.113 | 7.193 | 2.833 | 2.86× |
| kerr05_escape | 12.158 | 10.738 | 4.157 | 2.92× |

Three repeats, identical ray grids and physical settings: rtol/atol=1e-10/1e-12, max step 0.04 for the real finite-boundary ray and 0.3 for analytical-grid strong controls. Camera initialization is outside these trace timings. All statuses agree, no fallback was required, and raw RHS counts/null residuals are retained. The local exact-metric controls are imported Cartesian grids, not direct analytical shortcuts. Data activation/native validation, Python RHS/stepping and dense-output constraint audits explain why geodesic gain is below the metric gain.

## Rendering, actual worker counts and memory

| 16-ray grid render | Python s | NumPy s | Numba s |
|---|---:|---:|---:|
| Serial | 21.617 | 19.974 | 8.883 |
| Requested 2; actual 1 spawned child | 22.991 | 21.113 | 11.194 |

These include image assembly/output. The serial render improves about 2.43×. Normal production batching uses 64 rays: this 16-ray scene has only one task, so requesting two does not exercise two-core scaling. The earlier Phase 1 12-ray statement is corrected accordingly; the 432-ray preview did have enough tasks for two children. The old ≈345 s / 432-ray preview was not rerun or extrapolated.

A separate benchmark reuses the existing scheduler with **benchmark-only four-ray batches**, verifies two actual children and exactly equal full per-ray values/diagnostics, and leaves production scheduling untouched.

| Actual parallel tracing, 16 rays | Serial median s | Two-process median s | Speedup | Two-process range s | Worker peak RSS MiB |
|---|---:|---:|---:|---:|---:|
| python | 21.436 | 14.014 | 1.53× | 13.994–14.196 | 96.3 |
| numpy | 19.805 | 13.041 | 1.52× | 12.979–13.306 | 96.5 |
| numba | 9.010 | 8.738 | 1.03× | 8.674–9.143 | 163.0 |

For this tiny accelerated workload the second process gives only about 3%, within modest timing variability; use one worker for quick checks. This does not establish an optimal count for large images, ten physical cores or Hyper-Threading. Larger scenes need a separate scaling study. Worker startup, independent NumPy/Numba runtimes and global native validation are included. Pool-ready timing precedes lazy JIT initialization; the first task pays that cost.

| Process/mapping measurement | Python | NumPy | Numba (cache available) |
|---|---:|---:|---:|
| Constructor seconds | 0.8483 | 0.8598 | 0.8601 |
| First lookup seconds (includes mapping/validation/runtime/cache load) | 0.0478 | 0.0492 | 0.8667 |
| RSS after first lookup, MiB | 93.5 | 93.6 | 159.4 |
| Lifetime peak after full benchmark/profiling, MiB | 97.6 | 97.7 | 167.7 |
| Serialized warmed provider, bytes | 17009 | 16663 | 16663 |

The active pair maps 17,249,760 logical bytes (16.45 MiB); payloads are not copied into worker pickles. OS file-backed pages may be shared. RSS includes runtime/code/scratch and shared pages, so worker RSS sums are not unique physical memory. Tracemalloc peaks measure only tracked Python/NumPy allocations, not all native/JIT allocations.

## Initialization isolation

| Operation, warm OS file cache | Median s |
|---|---:|
| metadata | 0.041129 |
| hierarchy_constructor | 0.838680 |
| mmap_open_no_touch | 0.003636 |
| native_validation_warm_pages | 0.037761 |
| metadata_map_validate | 0.085288 |

The hierarchy constructor includes metadata/certification, not payload validation. Mmap opening is lazy, not a disk-bandwidth measurement; physical validation touches all native nodes. These costs overlap and must not be summed as independent phases. Global native validation and certification were not weakened.

With a genuinely fresh JIT cache, first lookup took **13.47 s**, including native mapping/validation, Numba import/runtime and compilation. RSS then was 274.6 MiB; lifetime peak was 276.4 MiB. That one cold measurement is not steady-state throughput and does not isolate compiler CPU time by subtraction. The later benchmark-only reconstruction compilation/profiling adds further peak memory.

## Numerical equivalence and rejected prototype

- linear: 1024 events, 1024 valid, 0 rejected; inverse/derivative maxima 0/0. Decisions and selected levels agree.
- cubic: 1024 events, 379 valid, 645 rejected; inverse/derivative maxima 0/0. Decisions and selected levels agree.
- cubic_amr_c1: 1024 events, 896 valid, 128 rejected; inverse/derivative maxima 0/0. Decisions and selected levels agree.
- cubic_amr_c1_box: 1064 events, 1064 valid, 0 rejected; inverse/derivative maxima 0/0. Decisions and selected levels agree.
- cubic_amr_c1_hierarchy: 1104 events, 1104 valid, 0 rejected; inverse/derivative maxima 0/0. Decisions and selected levels agree.
- cubic_amr_c1_hierarchy / real data: 1104 events, 1104 valid, 0 rejected; inverse/derivative maxima 0/0. Decisions and selected levels agree.

The final event set includes random off-grid points, exact/one-ULP faces, selected edges/corners and temporal endpoints. Unit tests additionally cover corrupt native data, lapse/SPD failures, cubic overshoot, ill-conditioned gamma, underflow, seams, missing stencils, overlap precedence, one-sided knots, map invalidation, initial conditions and serial/spawn equality. The final hybrid strong-ray comparison retained rtol=1e-11/1e-13 and the unchanged 2e-8 state agreement limit. Saved matched-affine controls at two tolerance settings have zero measured state difference on this machine (`accuracy-final/`). Bitwise equality is an observation on these workloads, not a guarantee across different BLAS/CPU/compiler builds.

Independent live C++ comparison: all 160 manufactured event classifications agree; maximum component difference 4.92e-16. The real-data maximum is 2.53e-15; common-state endpoint difference 1.43e-10, sky-direction difference 1.4e-11 rad. The C++ source and existing saved-golden regression thresholds are unchanged. The manufactured C++/Python trajectory crosses all three refinement levels and a temporal knot, ending at its intentional affine limit as unresolved.

**Rejected prototype:** scalar/fully compiled 3+1 reconstruction produced only few-ULP pointwise changes, but those altered adaptive histories on the strong imported Kerr/Schwarzschild cubes. At rtol=1e-11 state discrepancies reached roughly 1e-6–2e-6, exceeding the fixed 2e-8 check; tightening alone did not reliably meet it for every Kerr control. Those results remain in `accuracy/`, `accuracy-blas/`, `accuracy-tight/` and `accuracy-step/`. The prototype is not selectable for production; its isolated timing is explicitly labeled in benchmark JSON. Production instead combines compiled ADM interpolation with validated NumPy/LAPACK reconstruction; the strong controls then pass at their original tolerances. No acceptance threshold was relaxed.

## Regression and limits

Final complete suite: **290 passed in 559.84 s**. The full log is `test_results.txt`. All prior analytical/numerical tests remain intact.

Remaining opportunities are fewer RHS/audit allocations, preserving exact NumPy reconstruction while reducing its dispatch cost, amortizing native validation on repeated immutable remaps with appropriate change detection, and measuring larger-task multiprocessing. Beyond the explicitly optional Numba compiler, no GPU framework, physical approximation, numerical method, interpolation order, opacity model or integration tolerance was introduced. The supported data remain static rectangular AMR; native-domain/capture/background limitations from Phase 1 still apply.

See [README.md](README.md) for Windows/Ubuntu commands, dependency selection and reproduction. Raw timings, profiles, memory snapshots, source hashes, query hashes, accuracy arrays and worker provenance are preserved beside this report. No Git commands were used.
