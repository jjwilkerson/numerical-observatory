"""Regenerate the scaling report from completed measurement records."""
import argparse,json,statistics
from pathlib import Path


def median(values):return statistics.median(values)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--directory',type=Path,default=Path(__file__).resolve().parent)
    a=p.parse_args();root=a.directory;study=json.loads((root/'measured/study.json').read_text());runs=study['runs']
    groups={w:[r['record'] for r in runs if r['workers']==w] for w in (1,2,5,10)}
    if any(len(v)!=2 for v in groups.values()):raise RuntimeError('two measured repetitions per worker count are required')
    base=median([r['total_command_wall_seconds'] for r in groups[1]]);first=runs[0]['record']
    fields=('config_sha256','production_sha256','array_sha256','counts','termination_reasons','rhs_evaluations','integrated_rays','metric_backend_fallbacks')
    for run in runs:
        record=run['record']
        if record['actual_compute_workers']!=run['workers']:raise AssertionError('worker count mismatch')
        if any(record[field]!=first[field] for field in fields):raise AssertionError('configuration, source or numerical result mismatch')
        if not run['equality']['all_saved_ray_arrays_exactly_equal'] or not run['equality']['image_identical']:raise AssertionError('output mismatch')
    lines=['# Numba numerical-spacetime multiprocessing scaling','',
        'Measurement date: 2026-11-01. Production algorithms, interpolation, scheduler, batch size, integration tolerances and C++ reference sources were not edited. '
        'Production Python source hashes are identical before/after every run and across all configurations.','',
        f"Hardware/software: {first['machine']['cpu']}, 10 physical / 20 allowed logical CPUs, Windows 11, Python 3.13.7, NumPy 2.5.4, SciPy 1.18.1 and Numba 0.67.0. "
        'The same existing Numba disk cache was used throughout.','',
        '## Scene and method','',
        'Real three-level CarpetX GMDS dataset; `cubic_amr_c1_hierarchy`, `backend="numba"`, static observer at t=4, (6.5,0.23,0.17), looking along +X. '
        'Horizontal FOV 40°, 40×32 pixels, one ray per pixel, celestial-grid background, explicit finite direction sphere R=8, no capture surface. '
        'rtol=1e-10, atol=1e-12, max step=0.06, affine limit=6. The numerical observation is finite-boundary, not an asymptotic black-hole shadow.','',
        'The 32×24 pilot used five workers and took 138.35 s; its 12 batches were insufficient for multiple batches at ten workers. '
        'The final 40×32 grid has 1,280 rays and 20 unchanged 64-ray production batches. '
        'A final-scene ten-worker warm-up took 123.01 s; every worker completed exactly two batches. '
        'All kernels used the existing disk cache. The instrumented 4×4 prewarm exactly matched the prior saved production arrays.','',
        'Two measured repetitions ran sequentially in orders **1→2→5→10**, then **10→5→2→1**. '
        'No CPU-heavy Observatory tests or simulations ran concurrently. CPU frequency/temperature and unrelated system activity were not controlled. '
        'A fresh interpreter and fresh production worker pool were used for each run. No batch policy was altered.','',
        'The primary wall time is measured around the entire child command: interpreter/parent setup, cached-kernel preparation, worker startup, tracing, assembly, output, hashes and record writing. '
        'Render-only wall time and all other phase timings are retained separately. One worker means the normal serial parent (zero child processes); larger counts are verified active child workers.','',
        '## Scaling','',
        '| Workers | Actual workers | Median wall time | Speedup vs. 1 | Parallel efficiency | Peak memory |',
        '|---:|---:|---:|---:|---:|---:|']
    summary=[]
    for w,records in groups.items():
        wall=median([r['total_command_wall_seconds'] for r in records]);speed=base/wall
        peak=max(r['conservative_sum_of_process_peak_bytes'] for r in records)/2**20
        worker_peak=max((v['peak_memory'] for r in records for v in r['workers'].values()),default=max(r['parent_memory']['lifetime_peak_rss_bytes'] for r in records))/2**20
        lines.append(f'| {w} | {w} | {wall:.2f} s | {speed:.2f}× | {100*speed/w:.1f}% | ≤{peak:.0f} MiB summed RSS peaks; {worker_peak:.0f} MiB/compute process |')
        summary.append(dict(workers=w,median_wall_seconds=wall,speedup=speed,parallel_efficiency=speed/w,
            conservative_peak_sum_mib=peak,per_compute_process_peak_mib=worker_peak))
    best=min(summary,key=lambda r:r['median_wall_seconds'])
    lines+=['',f'**Recommendation for this tested scene: {best["workers"]} workers**, based on the lowest measured median wall time. '
        'This recommendation does not establish an optimum for larger datasets, different ray costs, more RAM, or different task granularity. '
        'Two repetitions describe observed variability, not a statistical confidence interval.','',
        'Memory is a conservative sum of each process’s individual peak RSS, not a measured simultaneous unique-memory peak. '
        'Shared libraries and file-backed mappings can be counted repeatedly. Native payloads remain read-only memory maps and are not serialized as full arrays.','',
        '## Individual timings and initialization','',
        '| Workers | Total wall repetitions, s | Median render wall, s | Parent provider + cached-kernel setup, s | Pool-ready span, s | First metric lookup per child, s |',
        '|---:|---:|---:|---:|---:|---:|']
    for w,records in groups.items():
        walls=', '.join(f'{r["total_command_wall_seconds"]:.2f}' for r in records)
        render=median([r['render_wall_seconds'] for r in records]);setup=median([r['parent_provider_constructor_seconds']+r['parent_first_metric_seconds'] for r in records])
        pool=median([r['execution'].get('startup_span_seconds',0.) for r in records])
        metric=[v['first_metric_seconds'] for r in records for v in r['workers'].values()]
        metric_text=f'{min(metric):.2f}–{max(metric):.2f}' if metric else 'parent only'
        lines.append(f'| {w} | {walls} | {render:.2f} | {setup:.2f} | {pool:.2f} | {metric_text} |')
    lines+=['','Pool-ready span measures spawn/unpickling/core imports before the existing initializer announces readiness. '
        'Lazy Numba runtime/cache loading, typed-list construction and the initial native-snapshot validation occur at first metric use inside the first task. '
        'Those times are measured once per process by a pass-through observer; they overlap between workers and must not be summed/subtracted as a global critical-path time. '
        'Later snapshot mapping/validation remains part of real ray-tracing work. No root-kernel cache misses were observed in measured runs.','',
        '## Work, CPU use and numerical equivalence','',
        '| Workers | Completed scheduled batches | Per-child batch range | Approx. aggregate CPU time, s | Average busy logical CPUs | Task wall sum less first metric, s |',
        '|---:|---:|---:|---:|---:|---:|']
    for w,records in groups.items():
        batch=sorted({r['completed_task_batches'] for r in records});counts=[v['task_batches'] for r in records for v in r['workers'].values()]
        count_text=f'{min(counts)}–{max(counts)}' if counts else 'serial, no scheduler tasks'
        cpu=median([r['approx_aggregate_cpu_seconds'] for r in records]);busy=median([r['average_busy_logical_cpus'] for r in records])
        pure=[sum(v['ray_task_seconds_less_first_metric'] for v in r['workers'].values()) for r in records]
        task=f'{median(pure):.2f}' if w>1 else 'not separately observable'
        lines.append(f'| {w} | {"/".join(map(str,batch))} | {count_text} | {cpu:.2f} | {busy:.2f} | {task} |')
    lines+=['','The serial renderer does not use scheduled batches; its 0 actual batches correspond to the same 20 batch-equivalents of ray work. '
        'Aggregate CPU time approximates parent lifetime CPU plus the final sampled lifetime CPU of each child; small shutdown/record-writing tails are excluded. '
        'Busy logical CPUs = aggregate CPU / total wall time (1.0 = one fully busy logical CPU), not physical-core utilization. '
        'Worker task-wall sums are aggregate work across concurrent processes and may exceed wall time.','',
        f'Every measured run traced **{first["integrated_rays"]:,} rays** with **{first["rhs_evaluations"]:,} RHS evaluations**. '
        f'Classifications: `{first["counts"]}`. Termination reasons: `{first["termination_reasons"]}`. '
        f'Backend fallback count: **{first["metric_backend_fallbacks"]}**. '
        'All saved per-ray fields, null residuals, statuses and classifications were exactly equal across all eight runs, as were rendered PNGs. '
        'Per-array hashes include dtype, shape and contiguous data bytes; full NPZ/PNG/diagnostic file hashes are also retained. '
        'Execution-specific diagnostic hashes may differ because timing/process provenance is included. No numerical tolerances were relaxed and no failures were omitted.','',
        '## Interpretation and limits','',
        'Two, five and ten workers completed exactly ten, four and two batches per child respectively, so there is no gross worker underuse or batch-count imbalance. '
        'Equal batch counts do not imply equal costs. Five→ten workers reduced median wall time by about 31.6%, rather than an ideal 50%. '
        'The pool-ready span grew from about 9.4 to 20.0 s, and aggregate CPU time increased from about 822 to 990 s for the same work. '
        'Initialization and increased per-work CPU cost are therefore visible costs; this is not evidence of a changed numerical workload.','',
        'Use the measured wall time, CPU-equivalent use, task distributions and phase costs together. '
        'Initialization cannot be assumed negligible merely because kernels are cached: each spawned process loads its own runtime and validates its first active snapshot pair. '
        'Two 64-ray tasks per process at ten workers limit dynamic load balancing; the final completion tail reflects the slowest task wave. '
        'Shared memory bandwidth, reduced all-core clock rates, OS activity and runtime overhead can reduce efficiency; these data do not independently identify a unique hardware bottleneck. '
        'There is no evidence here that Hyper-Threading doubles capacity, and 20 workers were not tested.','',
        '## Audit and reproduction','',
        'The measurement harness wraps only the first metric lookup (then restores the original method) and completed production tasks. '
        'It does not alter results, task sizes, scheduling bounds, worker initialization, inputs, tolerances or algorithms. '
        'One small JSON event is appended per task outside the original task timer; total wall includes that observational I/O. '
        'All production-source hashes match before/after each run. The C++ reference and full regression suite were not modified or rerun.','',
        'Windows PowerShell, from the repository root:',
        '```powershell',
        r'.\.venv\Scripts\python examples/numerical_optimization/scaling_numba_2026_11_01/benchmark_scaling.py --study --config examples/numerical_optimization/scaling_numba_2026_11_01/scene.toml --output output/numba-scaling-repeat',
        r'.\.venv\Scripts\python examples/numerical_optimization/scaling_numba_2026_11_01/report_scaling.py',
        '```',
        'On Ubuntu substitute `.venv/bin/python`. Use a fresh output directory; prior measurements are never overwritten. '
        'Run the harness first with `--launch --config examples/numerical_optimization/scaling_numba_2026_11_01/scene.toml --workers 10 --output output/numba-scaling-warmup` before a new study. '
        'The existing Numba cache must be available; the study rejects measured root-kernel cache misses.','',
        '`scene.toml`, pilot/prewarm configurations, raw worker JSONL observations, all render products, per-run records, output hashes and rotated order are archived here. '
        '`measured/study.json` is the complete machine-readable measurement manifest. No Git commands or CarpetX simulations were run.','']
    (root/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    (root/'summary.json').write_text(json.dumps(dict(scene=study['config'],config_sha256=study['config_sha256'],recommended_workers=best['workers'],results=summary),indent=2)+'\n')


if __name__=='__main__':main()
