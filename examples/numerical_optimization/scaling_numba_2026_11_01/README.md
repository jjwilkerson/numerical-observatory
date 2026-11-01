# Numba scaling study

[REPORT.md](REPORT.md) contains the measured results, initialization/CPU/memory
breakdown, numerical-equivalence audit and limitations. The tested 40×32 scene
was fastest with **10 workers**: median **120.75 s**, versus **729.01 s** serial
(**6.04×** speedup, **60.4%** parallel efficiency).

All eight measured runs produced identical per-ray arrays and PNGs, with 1,280
escaping rays, zero unresolved rays, zero fallbacks and 1,449,119 RHS evaluations
per run. Production files and scheduling were not modified.

Render this same scene with the recommended count:

```powershell
.\.venv\Scripts\python -m numerical_observatory --config examples/numerical_optimization/scaling_numba_2026_11_01/scene.toml --workers 10 --output output/numba-scaling-scene
```

On Ubuntu use `.venv/bin/python`. This recommendation is specific to the tested
scene, dataset and hardware, not a universal worker-count rule.

Artifacts:

- `scene.toml`: frozen comparison scene; `pilot.toml` and `prewarm.toml`: warm-ups.
- `measured/study.json`: all repetitions, order, raw timing records and hashes.
- `measured/repeat*/record.json`: phase/CPU/memory counts for each measured run.
- Each run's `observations/`: first-lookup and per-task JSONL measurements.
- Each run's `render/`: unchanged production render products and ray arrays.
- `summary.json`: medians, speedups, efficiencies and memory upper bounds.
- `audit.json`: source-preservation, worker-count and equality conclusions.
- `benchmark_scaling.py`: pass-through measurement harness; `report_scaling.py`:
  report generator. See the report for reproduction commands.

Peak-memory totals conservatively sum individual RSS peaks and can double-count
shared mappings/libraries. Two repetitions were performed; timings are not a
statistical confidence interval. No full regression suite, C++ changes, Git
commands or CarpetX simulations were used for this measurement-only task.
