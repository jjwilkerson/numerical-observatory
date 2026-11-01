"""Event and matched-affine accuracy comparisons, separate from benchmarks."""
import argparse,json,time
from pathlib import Path
import numpy as np
from .numerical_benchmark import make_case,queries,machine
from .numerical_metric import NumericalMetric,MODES
from .geodesic import Integration,trace
from .camera import Camera
from .kerr_reference import pixel_momentum


def angle(a,b):return float(np.arctan2(np.linalg.norm(np.cross(a,b)),np.dot(a,b)))


def event_comparisons(root,backend,count,real):
    rows=[]
    cases=[(mode,None) for mode in MODES]
    if real is not None:cases.append(('cubic_amr_c1_hierarchy',real))
    for mode,dataset in cases:
        ref,_=make_case(root/'fixtures',mode,'python',dataset);fast,_=make_case(root/'fixtures',mode,backend,dataset)
        events=list(queries(count,dataset is not None))
        for b in ref.bands:
            for bounds in (b.outer_lower,b.inner_lower,b.inner_upper,b.outer_upper):
                for d in range(3):
                    for direction in (-np.inf,None,np.inf):
                        xyz=(np.asarray(b.outer_lower)+b.outer_upper)/2
                        xyz[d]=bounds[d] if direction is None else np.nextafter(bounds[d],direction)
                        events.append(np.r_[4. if dataset is not None else .37,xyz])
                events.append(np.r_[4. if dataset is not None else .37,bounds])
        largest=np.zeros(2);squared=np.zeros(2);values=0;rejected=0
        for q in events:
            try:a=ref.evaluate(q)
            except ValueError as exc:
                try:fast.evaluate(q)
                except ValueError as other:
                    if exc.code!=other.code:raise AssertionError('different rejection reason')
                else:raise AssertionError('optimized evaluator accepted invalid query')
                rejected+=1;continue
            b=fast.evaluate(q)
            if ref.last_levels!=fast.last_levels:raise AssertionError('different selected levels')
            for i,(x,y) in enumerate(zip(a,b)):
                delta=x-y;largest[i]=max(largest[i],float(np.max(abs(delta))));squared[i]+=float(np.mean(delta**2))
            values+=1
        if max(largest)>2e-12:raise AssertionError('event comparison exceeded 2e-12')
        rows.append(dict(mode=mode,real=dataset is not None,valid=values,rejected=rejected,events=len(events),
            max_inverse_error=largest[0],max_derivative_error=largest[1],rms_errors=np.sqrt(squared/values).tolist(),
            classifications_and_levels_equal=True,backend_fallbacks=fast.backend_fallbacks))
    return rows


def strong_comparisons(root,backend,tolerances,offsets,max_step=.3,models=('schwarzschild','kerr05')):
    from .numerical_validation import ROOT
    base=ROOT/'examples/numerical_spacetime/validation';rows=[]
    for model in models:
        path=base/'synthetic'/f'{model}-shadow-0.5'
        if not path.exists():continue
        radius=next(r['reference_radius_pixels'] for r in json.loads((base/'shadow_boundary_convergence.json').read_text()) if r['model']==model)
        c=Camera(position=(12,0,0),target=(0,0,0),up=(0,0,1),time=30,width=96,height=72)
        domain=dict(background='finite_direction_sphere',background_radius=14.,capture_spheres=[dict(center=[0,0,0],radius=1.99 if model=='schwarzschild' else 1.7)])
        for offset in offsets:
            for tolerance in tolerances:
                ref=NumericalMetric(path,'cubic',domain=domain);fast=NumericalMetric(path,'cubic',domain=domain,backend=backend)
                p=pixel_momentum(ref,c,47.5+radius+offset,35.5)
                settings=Integration(rtol=tolerance,atol=tolerance/100,max_step=max_step,max_affine=150)
                start=time.perf_counter();a,s=trace(ref,c.event,p,settings,return_solution=True);ref_time=time.perf_counter()-start
                start=time.perf_counter();b,t=trace(fast,c.event,p,settings,return_solution=True);fast_time=time.perf_counter()-start
                sample=np.linspace(0,min(s.t[-1],t.t[-1]),151);state_error=np.max(abs(s.sol(sample)-t.sol(sample)),axis=1)
                np.savez_compressed(root/f'{model}-{offset}-{tolerance}.npz',affine=sample,
                    python=s.sol(sample),optimized=t.sol(sample),python_endpoint=s.y[:,-1],optimized_endpoint=t.y[:,-1])
                field_error=np.zeros(2)
                for q in s.sol(sample)[:4].T:
                    for i,(g,h) in enumerate(zip(ref.evaluate(q),fast.evaluate(q))):field_error[i]=max(field_error[i],float(np.max(abs(g-h))))
                row=dict(model=model,offset=offset,rtol=tolerance,max_step=max_step,python_status=a.status,optimized_status=b.status,
                    python_reason=a.reason,optimized_reason=b.reason,max_event_error=float(max(state_error[:4])),max_momentum_error=float(max(state_error[4:])),
                    max_inverse_error=field_error[0],max_derivative_error=field_error[1],python_null=a.max_null_error,optimized_null=b.max_null_error,
                    python_rhs=a.evaluations,optimized_rhs=b.evaluations,endpoint_error=float(max(abs(s.y[:,-1]-t.y[:,-1]))),
                    direction_error=None if a.direction is None or b.direction is None else angle(a.direction,b.direction),python_seconds=ref_time,optimized_seconds=fast_time)
                rows.append(row);(root/'strong_convergence.json').write_text(json.dumps(rows,indent=2)+'\n')
                print(model,offset,tolerance,'state error',max(state_error),flush=True)
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--backend',choices=('numpy','numba'),default='numba');p.add_argument('--count',type=int,default=512)
    p.add_argument('--real',type=Path);p.add_argument('--strong',action='store_true');p.add_argument('--events',action='store_true')
    p.add_argument('--tolerances',nargs='+',type=float,default=[1e-11,1e-12,1e-13]);p.add_argument('--offsets',nargs='+',type=float,default=[-.5,.5])
    p.add_argument('--max-step',type=float,default=.3);p.add_argument('--models',nargs='+',default=['schwarzschild','kerr05'])
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    if a.events:
        result=dict(machine=machine(),backend=a.backend,comparisons=event_comparisons(a.output,a.backend,a.count,a.real))
        (a.output/'event_comparisons.json').write_text(json.dumps(result,indent=2)+'\n')
    if a.strong:strong_comparisons(a.output,a.backend,a.tolerances,a.offsets,a.max_step,a.models)


if __name__=='__main__':main()
