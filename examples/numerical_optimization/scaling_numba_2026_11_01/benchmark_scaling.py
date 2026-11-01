"""Measurement-only wrapper around the unchanged production renderer/scheduler.

The first metric lookup and completed tasks are observed, never substituted.
Use --single for one fresh-interpreter render, or --compare to audit saved runs.
"""
import argparse,hashlib,json,os,statistics,subprocess,sys,time
from pathlib import Path
import numpy as np
from numerical_observatory import parallel
from numerical_observatory.numerical_metric import NumericalMetric
from numerical_observatory.cli import load_configuration
from numerical_observatory.background import CelestialGrid
from numerical_observatory.render import render
from numerical_observatory.numerical_benchmark import machine,memory

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
ORIGINAL_EVALUATE=NumericalMetric.evaluate
ORIGINAL_RUN=parallel._run


def emit(value):
    directory=os.environ.get('OBSERVATORY_SCALING_OBSERVATIONS')
    if directory:
        path=Path(directory)/f'{os.getpid()}.jsonl'
        with path.open('a',encoding='utf-8') as f:f.write(json.dumps(value,allow_nan=False)+'\n')


def first_evaluate(self,event,slab=None):
    # Restore before the call: subsequent evaluations have NO wrapper overhead.
    NumericalMetric.evaluate=ORIGINAL_EVALUATE
    start=time.perf_counter();cpu=time.process_time()
    try:return ORIGINAL_EVALUATE(self,event,slab)
    finally:
        seconds=time.perf_counter()-start;cpu_seconds=time.process_time()-cpu
        kernel=self._accelerator.kernels.adm if self._accelerator is not None else None
        hits=sum(getattr(kernel,'_cache_hits',{}).values()) if kernel else None
        misses=sum(getattr(kernel,'_cache_misses',{}).values()) if kernel else None
        emit(dict(kind='first_metric',pid=os.getpid(),seconds=seconds,cpu_seconds=cpu_seconds,
                  root_kernel_cache_hits=hits,root_kernel_cache_misses=misses,memory=memory()))


def observed_run(task):
    result=ORIGINAL_RUN(task)
    measured=result[2]
    emit(dict(kind='task',pid=os.getpid(),task_first=int(task[0]),rays=len(task),
              task_wall=measured['wall'],task_cpu=measured['cpu'],peak_memory=measured['peak_memory'],
              process_cpu_lifetime=time.process_time(),finished_monotonic=time.perf_counter()))
    return result


# Spawn imports this file too. These wrappers do not change return values,
# task sizes, initializers, in-flight limits, tolerances or scene contents.
NumericalMetric.evaluate=first_evaluate
parallel._run=observed_run


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def production_hashes():
    return {p.relative_to(ROOT).as_posix():sha(p) for p in sorted((ROOT/'src/numerical_observatory').glob('*.py'))}


def array_hash(array):
    value=np.ascontiguousarray(array)
    return hashlib.sha256(str(value.dtype).encode()+str(value.shape).encode()+value.tobytes()).hexdigest()


def single(config,workers,output):
    output=Path(output).resolve()
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True);observations=output/'observations';observations.mkdir()
    os.environ['OBSERVATORY_SCALING_OBSERVATIONS']=str(observations)
    before=production_hashes();harness_hash=sha(__file__);pipeline_start=time.perf_counter();cpu_start=time.process_time()
    t=time.perf_counter();metric,camera,settings=load_configuration(config);construct=time.perf_counter()-t
    t=time.perf_counter();metric.evaluate(camera.event);prepare=time.perf_counter()-t
    progress_rows=[];batch_progress=[];last=[0.]
    def progress(done,total):
        progress_rows.append([time.perf_counter(),done,total])
        if done==total or time.perf_counter()-last[0]>=30:
            print(f'workers={workers} completed rows {done}/{total}',flush=True);last[0]=time.perf_counter()
    def work(done,total,rays,total_rays):
        batch_progress.append([time.perf_counter(),done,total,rays,total_rays])
    progress.work=work
    t=time.perf_counter()
    report=render(metric,camera,settings,output/'render',progress=progress,background=CelestialGrid(),workers=workers,samples=1)
    render_wall=time.perf_counter()-t;pipeline_wall=time.perf_counter()-pipeline_start
    parent_cpu=time.process_time()-cpu_start;parent_lifetime=time.process_time()
    events=[json.loads(line) for p in observations.glob('*.jsonl') for line in p.read_text().splitlines()]
    tasks=[e for e in events if e['kind']=='task'];initial=[e for e in events if e['kind']=='first_metric']
    worker_pids=sorted({e['pid'] for e in tasks});execution=report.get('execution',{})
    launched=execution.get('workers',0);actual=1 if workers==1 else launched
    if workers>1 and (launched!=workers or len(worker_pids)!=workers):raise AssertionError('requested worker count was not fully active')
    by_worker={str(pid):dict(task_batches=sum(e['pid']==pid for e in tasks),
        rays=sum(e['rays'] for e in tasks if e['pid']==pid),
        task_wall=sum(e['task_wall'] for e in tasks if e['pid']==pid),
        task_cpu=sum(e['task_cpu'] for e in tasks if e['pid']==pid),
        process_cpu_lifetime=max(e['process_cpu_lifetime'] for e in tasks if e['pid']==pid),
        peak_memory=max(e['peak_memory'] for e in tasks if e['pid']==pid)) for pid in worker_pids}
    init_by_pid={str(e['pid']):e for e in initial}
    for pid,row in by_worker.items():
        first=init_by_pid.get(pid,{})
        row['first_metric_seconds']=first.get('seconds');row['first_metric_cpu']=first.get('cpu_seconds')
        row['ray_task_seconds_less_first_metric']=max(0.,row['task_wall']-first.get('seconds',0.))
    aggregate_cpu=parent_lifetime+sum(r['process_cpu_lifetime'] for r in by_worker.values())
    with np.load(output/'render/rays.npz') as rays:arrays={key:array_hash(rays[key]) for key in rays.files}
    after=production_hashes()
    if before!=after:raise AssertionError('production source changed during measurement')
    if sha(__file__)!=harness_hash:raise AssertionError('measurement harness changed during run')
    result=dict(config=str(Path(config).resolve()),config_sha256=sha(config),requested_workers=workers,actual_compute_workers=actual,
        child_processes_launched=launched,parent_pid=os.getpid(),resolution=[camera.width,camera.height],samples=1,
        equivalent_64_ray_batches=(camera.width*camera.height+63)//64,completed_task_batches=len(tasks),
        parent_provider_constructor_seconds=construct,parent_first_metric_seconds=prepare,
        render_wall_seconds=render_wall,pipeline_wall_seconds=pipeline_wall,parent_cpu_delta=parent_cpu,
        parent_cpu_lifetime=parent_lifetime,approx_aggregate_cpu_seconds=aggregate_cpu,parent_memory=memory(),
        execution=execution,workers=by_worker,initial_metric_observations=initial,
        counts=report['counts'],termination_reasons=report['termination_reasons'],rhs_evaluations=report['rhs_evaluations'],
        integrated_rays=report['integrated_rays'],metric_backend_fallbacks=report['metric_backend_fallbacks'],
        maxima=report['maxima'],progress_rows=progress_rows,batch_progress=batch_progress,
        array_sha256=arrays,output_sha256={name:sha(output/'render'/name) for name in ('rays.npz','shadow.png','diagnostics.json')},
        production_sha256=before,harness_sha256=harness_hash,machine=machine(),metric=metric.description())
    (output/'record.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(f'DONE workers={workers}, actual={actual}, render={render_wall:.2f}s, pipeline={pipeline_wall:.2f}s, {report["counts"]}',flush=True)


def launch(config,workers,output):
    output=Path(output).resolve();output.parent.mkdir(parents=True,exist_ok=True)
    log=output.parent/(output.name+'.log');start=time.perf_counter()
    with log.open('w') as handle:
        subprocess.run([sys.executable,'-u',str(Path(__file__).resolve()),'--single','--config',str(Path(config).resolve()),
                        '--workers',str(workers),'--output',str(output)],stdout=handle,stderr=subprocess.STDOUT,check=True)
    elapsed=time.perf_counter()-start;result=json.loads((output/'record.json').read_text())
    result['total_command_wall_seconds']=elapsed
    result['average_busy_logical_cpus']=result['approx_aggregate_cpu_seconds']/elapsed
    result['cpu_utilization_allowed_logical_percent']=100*result['average_busy_logical_cpus']/result['machine']['topology']['logical']
    result['conservative_sum_of_process_peak_bytes']=result['parent_memory']['lifetime_peak_rss_bytes']+sum(r['peak_memory'] for r in result['workers'].values())
    (output/'record.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result


def compare(paths):
    paths=list(map(Path,paths));reference=paths[0]
    report=[]
    with np.load(reference/'render/rays.npz') as a:
        for path in paths:
            maxima={}
            with np.load(path/'render/rays.npz') as b:
                if set(a.files)!=set(b.files):raise AssertionError('different ray-data fields')
                for key in a.files:
                    np.testing.assert_array_equal(a[key],b[key])
                    maxima[key]=0.
            report.append(dict(run=str(path),all_saved_ray_arrays_exactly_equal=True,max_absolute_differences=maxima,
                               image_identical=sha(path/'render/shadow.png')==sha(reference/'render/shadow.png')))
    return report


def study(config,directory):
    directory=Path(directory).resolve();directory.mkdir(parents=True,exist_ok=True)
    manifest=directory/'study.json'
    if manifest.exists():raise FileExistsError(manifest)
    records=[];reference=None
    orders=[[1,2,5,10],[10,5,2,1]]
    for repetition,order in enumerate(orders,1):
        for workers in order:
            output=directory/f'repeat{repetition}-workers{workers}'
            print(f'START repetition={repetition}, workers={workers}',flush=True)
            result=launch(config,workers,output)
            if result['actual_compute_workers']!=workers:raise AssertionError('actual worker mismatch')
            if reference is None:reference=output
            equality=compare([reference,output])[-1]
            if not equality['image_identical']:raise AssertionError('rendered image changed across workers')
            if any(e['root_kernel_cache_misses'] for e in result['initial_metric_observations']):
                raise AssertionError('cold root-kernel compilation detected in measured run; raw output retained')
            record=dict(repetition=repetition,workers=workers,output=str(output),record=result,equality=equality)
            records.append(record)
            manifest.write_text(json.dumps(dict(orders=orders,config=str(Path(config).resolve()),config_sha256=sha(config),runs=records),indent=2,allow_nan=False)+'\n')
            print(f'FINISH repetition={repetition}, workers={workers}, total={result["total_command_wall_seconds"]:.2f}s; arrays/image exactly equal',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--single',action='store_true')
    p.add_argument('--config',type=Path);p.add_argument('--workers',type=int,default=1);p.add_argument('--output',type=Path)
    p.add_argument('--launch',action='store_true');p.add_argument('--compare',nargs='+',type=Path)
    p.add_argument('--study',action='store_true')
    a=p.parse_args()
    if a.compare:
        value=compare(a.compare)
        if a.output:a.output.write_text(json.dumps(value,indent=2)+'\n')
        else:print(json.dumps(value,indent=2))
    elif a.study:study(a.config,a.output)
    elif a.single:single(a.config,a.workers,a.output)
    elif a.launch:
        result=launch(a.config,a.workers,a.output);print(json.dumps({k:result[k] for k in ('total_command_wall_seconds','actual_compute_workers','counts','completed_task_batches')},indent=2))
    else:p.error('choose --single, --launch, --study, or --compare')


if __name__=='__main__':main()
