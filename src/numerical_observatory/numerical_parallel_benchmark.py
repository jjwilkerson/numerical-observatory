"""True one/two-process comparison using existing scheduler and small ray batches.

The normal renderer batches 64 rays, so a 16-ray render requests only one actual
child even with --workers 2. These benchmark-only four-ray tasks exercise two
children without changing production scheduling or launching larger renders.
"""
import argparse,json,time,statistics
from pathlib import Path
import numpy as np
from .numerical_metric import NumericalMetric
from .numerical_benchmark import machine,memory
from .camera import Camera
from .geodesic import Integration
from .symmetry import CenteredRayCache
from .parallel import scheduled


def run(metric,camera,settings,workers):
    values={};stats={};size=camera.width*camera.height
    if workers==1:
        cache=CenteredRayCache(metric,camera,settings,enabled=False)
        for row,col,momentum in camera.rays(metric):
            values[row*camera.width+col]=cache.ray(row,col,momentum)
    else:
        tasks=[np.arange(i,min(i+4,size)) for i in range(0,size,4)]
        scene=('ode',metric,camera,settings,None,False,None)
        for batch,_ in scheduled(scene,tasks,workers,stats):
            for row,col,_,result in batch:values[row*camera.width+col]=result
        if stats['workers']!=workers:raise RuntimeError('requested parallelism was not actually exercised')
    return [values[i] for i in range(size)],stats


def summarize(results):
    return dict(status=[r.status for r in results],reasons=[r.reason for r in results],
        directions=[None if r.direction is None else r.direction.tolist() for r in results],
        null=[r.max_null_error for r in results],rhs=[r.evaluations for r in results],
        numerical=[r.numerical_diagnostics for r in results])


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--backend',choices=('python','numpy','numba'),required=True)
    p.add_argument('--repeats',type=int,default=3);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    c=Camera(position=(6.5,.23,.17),target=(7.5,.23,.17),up=(0,0,1),time=4,width=4,height=4,fov_degrees=40)
    settings=Integration(rtol=1e-10,atol=1e-12,max_step=.06,max_affine=6);measurements=[];reference=None
    for workers in (1,2):
        times=[];execution=[]
        for i in range(a.repeats):
            m=NumericalMetric(a.dataset,'cubic_amr_c1_hierarchy',backend=a.backend,domain=dict(background='finite_direction_sphere',background_radius=8.))
            m.evaluate(c.event);start=time.perf_counter();results,stats=run(m,c,settings,workers);times.append(time.perf_counter()-start)
            value=summarize(results)
            if reference is None:reference=value
            if reference!=value:raise AssertionError('serial/spawn numerical results or diagnostics differ')
            execution.append(stats)
        measurements.append(dict(workers=workers,seconds=times,median_seconds=statistics.median(times),
            stdev_seconds=statistics.stdev(times) if len(times)>1 else 0.,execution=execution,parent_memory=memory()))
        result=dict(machine=machine(),backend=a.backend,rays=16,task_rays=4,measurements=measurements,
                    all_ray_values_exactly_equal=True,ray_values=reference)
        if len(measurements)==2:result['speedup_two_workers']=measurements[0]['median_seconds']/measurements[1]['median_seconds']
        (a.output/'benchmark.json').write_text(json.dumps(result,indent=2)+'\n')
        print(a.backend,'actual workers',workers,'seconds',times,flush=True)


if __name__=='__main__':main()
