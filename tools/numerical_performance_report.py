"""Build the performance report from saved, auditable benchmark JSON."""
import argparse,json,re
from pathlib import Path


def load(path):return json.loads(Path(path).read_text())


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,default=Path('examples/numerical_optimization'))
    p.add_argument('--tests',type=Path,default=Path('output/numerical-optimization-complete-tests.txt'));a=p.parse_args();root=a.data
    backends=('python','numpy','numba');synthetic={b:load(root/'controlled-final'/b/'synthetic/benchmark.json') for b in backends}
    real={b:load(root/'controlled-final'/b/'real/benchmark.json') for b in backends}
    parallel={b:load(root/'parallel'/b/'benchmark.json') for b in backends};init=load(root/'initialization.json')
    cold=load(root/'cold-jit/benchmark.json')['cases'][0];accuracy=load(root/'accuracy-production/event_comparisons.json')
    cpp=load(root/'cpp-production/cpp_parity.json');native_cpp=load(root/'cpp-production/real_cpp_parity.json')
    preserved=load(root/'preserved_sources.json');assert all(preserved.values())
    lines=['# Numerical metric optimization: measured performance and validation','',
        'Measured 2026-11-01 on '+real['python']['machine']['cpu']+'. Windows 11, Python 3.13.7, NumPy 2.5.4, SciPy 1.18.1, Numba 0.67.0 / llvmlite 0.49.0. '
        'The topology probe found 10 physical / 20 allowed logical CPUs. Python production tests ran on Windows; the independent C++ oracle ran under Ubuntu/WSL. A full Linux Python regression run was not performed.','',
        '## Result and implementation','',
        'The optional `numba` backend compiles all five spatial interpolation modes and temporal ADM interpolation, while retaining NumPy/LAPACK reconstruction with validated broadcast derivatives. '
        '`python` remains the unchanged default/reference path; `numpy` offers a dependency-free smaller gain. No integration tolerance, camera, physical field convention, native-data validation, coverage certificate or geodesic equation was relaxed. '
        'There is no GPU, disk or dynamic-AMR implementation in this phase.','',
        'Production changes are confined to optional metric dispatch, cached immutable geometry/views, strict CPU kernels, equivalent temporal-index binary search, vectorized algebra and additional provenance/counters. '
        'The data loader, AMR coverage-certification module and nine analytical/camera/disk modules retain their prior source hashes (`preserved_sources.json`). '
        'The DOP853 stepping code is unchanged; its additions only record metric backend/fallback counts.','',
        '## Controlled method','',
        'The authoritative `controlled-final/` runs used sequential isolated interpreter processes, 512 deterministic distinct/stratified events, three repeats, and identical dataset/query hashes and valid-query sets across backends. '
        'No test suite or other Observatory compute job was launched alongside these benchmarks; unrelated system activity was not controlled. '
        'All repeat times, variability, CPU/software/source hashes and rejection counts are retained in JSON. OS page caches were not evicted. '
        'Early `baseline/`, `numpy/` and `controlled/` runs document stages; the final table below is the matched final comparison.','',
        '## Warm metric evaluations','',
        '| Workload | Python eval/s | NumPy eval/s | NumPy speedup | Numba eval/s | Numba speedup |',
        '|---|---:|---:|---:|---:|---:|']
    summary=dict(metric=[],trajectories=[],parallel=[])
    groups=[('synthetic',synthetic),('real',real)]
    for name,group in groups:
        for i,base in enumerate(group['python']['cases']):
            cases={b:group[b]['cases'][i] for b in backends}
            assert len({r['query_sha256'] for r in cases.values()})==1
            assert all(r['valid_events']==base['valid_events'] and r['rejections']==base['rejections'] for r in cases.values())
            assert all(group[b]['machine']['source_sha256']==group['python']['machine']['source_sha256'] for b in backends)
            rate={b:cases[b]['evaluations_per_second'] for b in backends};label=(name+' '+base['mode'])
            lines.append(f"| {label} | {rate['python']:,.0f} | {rate['numpy']:,.0f} | {rate['numpy']/rate['python']:.2f}× | {rate['numba']:,.0f} | {rate['numba']/rate['python']:.2f}× |")
            summary['metric'].append(dict(workload=label,rates=rate,numba_speedup=rate['numba']/rate['python']))
    lines+=['','The ≥5× target is met on the representative manufactured and real three-level hierarchy workloads; the ≥10× metric stretch target was not met. '
        'All five modes improve; the slowest relative compiled gain is still about 3.9×. Rejected requests are timed separately and consult the reference evaluator; their counts/reasons agree. '
        'These are complete metric-and-four-derivative calls, not repeated hits on the single-event memo.','',
        '| Real hierarchy repeat variability | Median loop seconds / 512 events | Standard deviation |', '|---|---:|---:|']
    for b in backends:
        t=real[b]['cases'][0]['metric'];lines.append(f"| {b} | {t['median_seconds']:.6f} | {t['stdev_seconds']:.6f} |")
    lines+=['','## Component costs and profiling','',
        '| Real hierarchy component | Python µs/operation | Selected compiled-path µs/operation |', '|---|---:|---:|']
    ref=real['python']['cases'][0];fast=real['numba']['cases'][0]
    for label,key,fastkey in [('Single-snapshot spatial interpolation','spatial','spatial'),('64-node gather','gather','backend_gather'),
            ('Native NumPy final-interpolant validity check','validation','validation'),('3+1 reconstruction','reconstruction','backend_reconstruction'),('Full bare Hamiltonian RHS','rhs','rhs')]:
        x=ref[key]['median_seconds']/ref['valid_events']*1e6;y=fast[fastkey]['median_seconds']/fast['valid_events']*1e6
        lines.append(f'| {label} | {x:.2f} | {y:.2f} |')
    lines+=['','The gather/stencil-construction microbenchmarks use a common 64-node stencil even for linear mode; its complete interpolation timing uses the correct eight nodes. '
        'Spatial timings include the reference-required independent/composite validity checks. The final NumPy validity check is deliberately retained. '
        'Component timings overlap and must not be summed. The bare-RHS workload uses an arbitrary fixed covector to measure all algebra; integrated rays use physically null camera initial conditions.','',
        'Initial profiling identified interpolation, node gathering, repeated small allocations and validation as the dominant costs. '
        'The first low-risk NumPy stage gave a modest gain. A fully compiled reconstruction prototype was faster but failed strong-ray equivalence (see below), so production keeps NumPy/LAPACK algebra. '
        'A separate 512-event reconstruction experiment found the broadcast NumPy version bitwise equal to the original and 1.75× faster (`reconstruction_choice.json`). '
        'Final profiling then identified scalar time-index clipping and repeated view bookkeeping; equivalent binary search and map-identity caching raised real throughput from about 5,012 to 6,017/s. '
        'New tests force exact/one-ULP knot queries and external map reactivation to verify invalidation.','',
        'After optimization, NumPy reconstruction/final validation and Python/kernel dispatch are prominent remaining costs. '
        'Compiled spatial interpolation alone is about 10.9× faster on this workload; that is **not** the complete metric or renderer speedup.','',
        '## End-to-end geodesics','',
        '| Control | Python s | NumPy s | Numba s | Numba speedup |', '|---|---:|---:|---:|---:|']
    for i,base in enumerate(real['python']['trajectories']):
        rows={b:real[b]['trajectories'][i] for b in backends};seconds={b:r['timing']['median_seconds'] for b,r in rows.items()}
        assert all(r['statuses']==base['statuses'] and r['reasons']==base['reasons'] for r in rows.values())
        lines.append(f"| {base['case']} | {seconds['python']:.3f} | {seconds['numpy']:.3f} | {seconds['numba']:.3f} | {seconds['python']/seconds['numba']:.2f}× |")
        summary['trajectories'].append(dict(case=base['case'],seconds=seconds,speedup=seconds['python']/seconds['numba']))
    lines+=['','Three repeats, identical ray grids and physical settings: rtol/atol=1e-10/1e-12, max step 0.04 for the real finite-boundary ray and 0.3 for analytical-grid strong controls. '
        'Camera initialization is outside these trace timings. All statuses agree, no fallback was required, and raw RHS counts/null residuals are retained. '
        'The local exact-metric controls are imported Cartesian grids, not direct analytical shortcuts. '
        'Data activation/native validation, Python RHS/stepping and dense-output constraint audits explain why geodesic gain is below the metric gain.','',
        '## Rendering, actual worker counts and memory','',
        '| 16-ray grid render | Python s | NumPy s | Numba s |', '|---|---:|---:|---:|']
    for i,label in [(0,'Serial'),(1,'Requested 2; actual 1 spawned child')]:
        values=[real[b]['renders'][i]['timing']['median_seconds'] for b in backends]
        if i==1:assert all(all(e['workers']==1 for e in real[b]['renders'][i]['execution']) for b in backends)
        lines.append('| '+label+' | '+' | '.join(f'{v:.3f}' for v in values)+' |')
    lines+=['','These include image assembly/output. The serial render improves about 2.43×. '
        'Normal production batching uses 64 rays: this 16-ray scene has only one task, so requesting two does not exercise two-core scaling. '
        'The earlier Phase 1 12-ray statement is corrected accordingly; the 432-ray preview did have enough tasks for two children. '
        'The old ≈345 s / 432-ray preview was not rerun or extrapolated.','',
        'A separate benchmark reuses the existing scheduler with **benchmark-only four-ray batches**, verifies two actual children and exactly equal full per-ray values/diagnostics, and leaves production scheduling untouched.','',
        '| Actual parallel tracing, 16 rays | Serial median s | Two-process median s | Speedup | Two-process range s | Worker peak RSS MiB |', '|---|---:|---:|---:|---:|---:|']
    for b in backends:
        q=parallel[b];s,t=q['measurements'];assert q['all_ray_values_exactly_equal']
        assert all(e['workers']==2 for e in t['execution'])
        peak=max(v for e in t['execution'] for v in e['worker_peak_memory_bytes'].values())/2**20
        lines.append(f"| {b} | {s['median_seconds']:.3f} | {t['median_seconds']:.3f} | {q['speedup_two_workers']:.2f}× | {min(t['seconds']):.3f}–{max(t['seconds']):.3f} | {peak:.1f} |")
        summary['parallel'].append(dict(backend=b,serial=s['median_seconds'],parallel=t['median_seconds'],speedup=q['speedup_two_workers']))
    lines+=['','For this tiny accelerated workload the second process gives only about 3%, within modest timing variability; use one worker for quick checks. '
        'This does not establish an optimal count for large images, ten physical cores or Hyper-Threading. Larger scenes need a separate scaling study. '
        'Worker startup, independent NumPy/Numba runtimes and global native validation are included. Pool-ready timing precedes lazy JIT initialization; the first task pays that cost.','',
        '| Process/mapping measurement | Python | NumPy | Numba (cache available) |', '|---|---:|---:|---:|']
    for label,key in [('Constructor seconds','constructor_seconds'),('First lookup seconds (includes mapping/validation/runtime/cache load)','first_lookup_seconds')]:
        lines.append('| '+label+' | '+' | '.join(f"{real[b]['cases'][0][key]:.4f}" for b in backends)+' |')
    lines.append('| RSS after first lookup, MiB | '+' | '.join(f"{real[b]['cases'][0]['memory_after_first_lookup']['rss_bytes']/2**20:.1f}" for b in backends)+' |')
    lines.append('| Lifetime peak after full benchmark/profiling, MiB | '+' | '.join(f"{real[b]['cases'][0]['process_memory_after']['lifetime_peak_rss_bytes']/2**20:.1f}" for b in backends)+' |')
    lines.append('| Serialized warmed provider, bytes | '+' | '.join(str(real[b]['cases'][0]['pickled_provider_bytes']) for b in backends)+' |')
    lines+=['',f"The active pair maps {ref['mapped_bytes']:,} logical bytes ({ref['mapped_bytes']/2**20:.2f} MiB); payloads are not copied into worker pickles. OS file-backed pages may be shared. "
        'RSS includes runtime/code/scratch and shared pages, so worker RSS sums are not unique physical memory. Tracemalloc peaks measure only tracked Python/NumPy allocations, not all native/JIT allocations.','',
        '## Initialization isolation','', '| Operation, warm OS file cache | Median s |', '|---|---:|']
    for key in ('metadata','hierarchy_constructor','mmap_open_no_touch','native_validation_warm_pages','metadata_map_validate'):
        lines.append(f"| {key} | {init[key]['median_seconds']:.6f} |")
    lines+=['','The hierarchy constructor includes metadata/certification, not payload validation. Mmap opening is lazy, not a disk-bandwidth measurement; physical validation touches all native nodes. '
        'These costs overlap and must not be summed as independent phases. Global native validation and certification were not weakened.','',
        f"With a genuinely fresh JIT cache, first lookup took **{cold['first_lookup_seconds']:.2f} s**, including native mapping/validation, Numba import/runtime and compilation. "
        f"RSS then was {cold['memory_after_first_lookup']['rss_bytes']/2**20:.1f} MiB; lifetime peak was {cold['memory_after_first_lookup']['lifetime_peak_rss_bytes']/2**20:.1f} MiB. "
        'That one cold measurement is not steady-state throughput and does not isolate compiler CPU time by subtraction. The later benchmark-only reconstruction compilation/profiling adds further peak memory.','',
        '## Numerical equivalence and rejected prototype','']
    for row in accuracy['comparisons']:
        lines.append(f"- {row['mode']}{' / real data' if row['real'] else ''}: {row['events']} events, {row['valid']} valid, {row['rejected']} rejected; inverse/derivative maxima {row['max_inverse_error']:.3g}/{row['max_derivative_error']:.3g}. Decisions and selected levels agree.")
    lines+=['','The final event set includes random off-grid points, exact/one-ULP faces, selected edges/corners and temporal endpoints. '
        'Unit tests additionally cover corrupt native data, lapse/SPD failures, cubic overshoot, ill-conditioned gamma, underflow, seams, missing stencils, overlap precedence, one-sided knots, map invalidation, initial conditions and serial/spawn equality. '
        'The final hybrid strong-ray comparison retained rtol=1e-11/1e-13 and the unchanged 2e-8 state agreement limit. Saved matched-affine controls at two tolerance settings have zero measured state difference on this machine (`accuracy-final/`). '
        'Bitwise equality is an observation on these workloads, not a guarantee across different BLAS/CPU/compiler builds.','',
        f"Independent live C++ comparison: all {sum(r['events'] for r in cpp['probes'])} manufactured event classifications agree; maximum component difference {max(r['max_component_difference'] for r in cpp['probes']):.3g}. "
        f"The real-data maximum is {native_cpp['max_metric_derivative_difference']:.3g}; common-state endpoint difference {native_cpp['endpoint_error']:.3g}, sky-direction difference {native_cpp['source_direction_error']:.3g} rad. "
        'The C++ source and existing saved-golden regression thresholds are unchanged. The manufactured C++/Python trajectory crosses all three refinement levels and a temporal knot, ending at its intentional affine limit as unresolved.','',
        '**Rejected prototype:** scalar/fully compiled 3+1 reconstruction produced only few-ULP pointwise changes, but those altered adaptive histories on the strong imported Kerr/Schwarzschild cubes. '
        'At rtol=1e-11 state discrepancies reached roughly 1e-6–2e-6, exceeding the fixed 2e-8 check; tightening alone did not reliably meet it for every Kerr control. '
        'Those results remain in `accuracy/`, `accuracy-blas/`, `accuracy-tight/` and `accuracy-step/`. '
        'The prototype is not selectable for production; its isolated timing is explicitly labeled in benchmark JSON. '
        'Production instead combines compiled ADM interpolation with validated NumPy/LAPACK reconstruction; the strong controls then pass at their original tolerances. No acceptance threshold was relaxed.','',
        '## Regression and limits','']
    text=a.tests.read_text() if a.tests.exists() else '';match=re.search(r'(\d+) passed in ([\d.]+)s',text)
    if match:lines.append(f"Final complete suite: **{match[1]} passed in {match[2]} s**. The full log is `test_results.txt`. All prior analytical/numerical tests remain intact.")
    else:lines.append('Final complete regression run is still in progress; update this report after it completes.')
    lines+=['','Remaining opportunities are fewer RHS/audit allocations, preserving exact NumPy reconstruction while reducing its dispatch cost, amortizing native validation on repeated immutable remaps with appropriate change detection, and measuring larger-task multiprocessing. '
        'Beyond the explicitly optional Numba compiler, no GPU framework, physical approximation, numerical method, interpolation order, opacity model or integration tolerance was introduced. '
        'The supported data remain static rectangular AMR; native-domain/capture/background limitations from Phase 1 still apply.','',
        'See [README.md](README.md) for Windows/Ubuntu commands, dependency selection and reproduction. '
        'Raw timings, profiles, memory snapshots, source hashes, query hashes, accuracy arrays and worker provenance are preserved beside this report. No Git commands were used.','']
    (root/'PERFORMANCE.md').write_text('\n'.join(lines),encoding='utf-8')
    (root/'performance_summary.json').write_text(json.dumps(summary,indent=2)+'\n')


if __name__=='__main__':main()
