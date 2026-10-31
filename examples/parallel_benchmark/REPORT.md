# Parallel CPU rendering benchmark

Recommendation for this computer: **10 workers** for expensive disk/ODE renders.
Automatic selection also identifies 10 physical cores. Keep one worker for small
quadrature renders. The 640×480 serial reference supplied by the user was about
one hour; it was not repeated. These controlled measurements support a worker
choice rather than an exact prediction of that full-size runtime.

## Method

Windows 11 (build 26200), Python 3.13.7, 10 available physical cores and 20 logical
CPUs, detected through Windows topology intersected with process affinity. Each
case runs in a fresh Python process, with identical scene, tolerances, emission,
palette and sampling at every worker count. No other benchmark runs concurrently.
All ODE scenes use the existing settings from `examples/disk_edge_on.toml`:
rtol=1e-9, atol=1e-11, max_step=4M, camera radius 60M, FOV 60°.

Wall time includes the render call, process startup, computation, parent shading,
assembly, compression and staged file publication. Parent interpreter startup
and initial scene/background loading are excluded. The script also prints
complete command wall time. Raw measurements are in [results.json](results.json)
and each case's `measurement.json` and `diagnostics.json`.

Scenes:

- Small background: 64×48, celestial grid, orbital quadrature, 364 distinct radii.
- Moderate disk: 128×96, 85° opaque disk and NASA starfield, thermal palette,
  12,288 rays and 1,338 distinct ODE trajectories. Serial integration lasts
  about 146 seconds, so this is substantially larger than startup overhead.
- Supersampled disk: 24×18 output, 2×2 samples (48×36 rays), grid, same disk.
- Rotational reuse: 48×36 output, background-only grid with general ODE tracing
  and exact centered Schwarzschild reuse.

## Wall time, speedup and efficiency

Entries after the serial column are **seconds / speedup / parallel efficiency**.
Speedup = serial wall / parallel wall; efficiency = speedup / requested workers.
All four worker counts were actually created in these completed benchmark cases.

| Scene | 1 worker, seconds | 5 workers | 10 workers | 20 workers |
|---|---:|---:|---:|---:|
| Small background quadrature | 0.094 | 2.207 / 0.042× / 0.8% | 3.072 / 0.030× / 0.3% | 5.089 / 0.018× / 0.1% |
| Moderate disk | 145.948 | 33.329 / 4.379× / 87.6% | 23.039 / 6.335× / 63.3% | 24.408 / 5.980× / 29.9% |
| Supersampled disk | 22.188 | 7.209 / 3.078× / 61.6% | 6.390 / 3.472× / 34.7% | 8.014 / 2.769× / 13.8% |
| Rotational reuse ODE | 13.672 | 5.291 / 2.584× / 51.7% | 4.924 / 2.777× / 27.8% | 6.960 / 1.964× / 9.8% |

Because 10 and 20 were close, each was repeated twice on the moderate disk, with
the order reversed for the final repeat:

| Workers | Run 1 | Run 2 | Run 3 | Median |
|---|---:|---:|---:|---:|
| 10 | 23.039 s | 22.142 s | 22.374 s | **22.374 s** |
| 20 | 24.408 s | 23.883 s | 24.040 s | **24.040 s** |

Twenty workers were 7.4% slower by median and used roughly twice the worker
memory. The median 10-worker speedup against the measured serial reference is
6.52×. The serial and 5-worker measurements were not repeated; OS scheduling,
CPU frequency and external workload can affect absolute timings.

## Memory, CPU accounting and overhead

Moderate-disk measurements, using the original runs:

| Workers | Parent peak working set | Sum of worker peaks | Combined peak upper bound | Busy logical-CPU equivalents |
|---|---:|---:|---:|---:|
| 1 | 517.5 MiB | 0 | 517.5 MiB | 1.00 |
| 5 | 516.8 MiB | 380.6 MiB | 897.4 MiB | 4.67 |
| 10 | 517.2 MiB | 757.1 MiB | 1,274.2 MiB | 8.65 |
| 20 | 517.4 MiB | 1,510.6 MiB | 2,028.0 MiB | 15.00 |

Memory uses OS process peak working-set counters; the sum of individual peaks
is an upper bound, not a simultaneous system peak. It includes the fresh
process's scene-loading peak even though loading time is excluded from wall
timing. Initialization samples account for idle workers as well as workers that
return tasks. Dense trajectories are kept only within a batch in each worker;
the sky texture is loaded only in the parent.

Busy logical-CPU equivalents = (parent render CPU seconds + worker task CPU
seconds) / wall seconds. Worker startup/import CPU time is excluded, making
this a lower bound on overall CPU use. Relative to 20 available logical CPUs,
these are approximately 5%, 23%, 43% and 75%. SMT CPU-time accounting counts
hardware threads; 15 busy logical CPUs do not mean 15 independent physical
cores. Windows CPU accounting can round very short quadrature tasks to zero.

An additional 128-empty-message test measures spawn/import/communication/shutdown
baseline costs: 1.603 s (one spawned child), 2.174 s (5), 2.883 s (10), and
5.032 s (20). Ordinary serial rendering spawns no child and pays none of that
pool startup cost. Once the last worker initialized, the remainder of these
empty-message tests was about 0.07–0.13 s; that includes messaging and shutdown,
not an isolated measurement of IPC. Real ray-result payload costs are included
in render wall times and parent CPU accounting. The moderate disk's startup
spans were 2.15 s, 2.87 s and 5.00 s for 5/10/20 workers.

## Correctness and lifecycle

[equality.json](equality.json) records exact comparisons across every scene and
1/5/10/20 workers: PNG file bytes, every saved ray-array byte (including subpixel
ordering, NaNs, hits, g, bolometric intensity, crossing indices and directions),
and all non-execution numerical diagnostics agree. No floating-point tolerance
was needed. Timings, process IDs, worker accounting and execution fields differ.
The integration count is identical at every worker count, including all 1,338
representative trajectories in the substantial disk test.

The automated suite additionally exercises 1/2/3 workers, disk-disabled paths,
all three palettes, both backgrounds, supersampling, off-center fallback,
spawn CLI startup, worker exceptions, abrupt process exit, process-creation
failure, interruption during a wait, and interruption during result assembly.
Failures leave no newly published render and preserve prior output files;
cleanup tests verify there are no remaining child processes. Physics tests and
earlier validated behavior remain part of the full suite.
Final complete run: **72 passed in 88.51 seconds** on Windows.

## Reproduction and limitations

```text
python examples/benchmark_parallel.py --repeat-selection
python -m pytest -q
```

Use `--resume` to reuse completed cases. For a 640×480 render in PowerShell:

```powershell
.\.venv\Scripts\python -m numerical_observatory --config examples/disk_edge_on.toml --width 640 --height 480 --workers 10 --output output/disk-edge-on-parallel
```

Parent-side shading and output, message transfer, load imbalance, process
startup and shared physical core resources limit scaling. The implementation
uses bounded dynamic tasks and preserves exact orbit reuse; increasing workers
does not increase the number of trajectories integrated. An external prefilled
dense path cache is supported only in serial mode. Custom parallel scenes must
be pickleable. Ubuntu uses the same explicit spawn path, with Linux affinity
and topology detection, but these hardware measurements were made on Windows.
CPU quotas without affinity restrictions require choosing workers explicitly.

Possible future work includes amortizing startup across a sequence of renders
and measuring whether parent display/output becomes the bottleneck at much
higher resolutions. No GPU backend, physical approximation or tolerance
relaxation is introduced here.
