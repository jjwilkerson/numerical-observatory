"""Optional modest numerical-grid experiments; never run by the render CLI."""
import argparse,json,time,pickle
from pathlib import Path
import numpy as np
from .numerical_validation import save
from .numerical_fixtures import write_dataset,adm_from_metric
from .numerical_metric import NumericalMetric
from .metric import Schwarzschild
from .kerr import Kerr
from .camera import Camera
from .geodesic import trace,Integration
from .kerr_reference import pixel_momentum,shadow_curve,reference_radius
from .binary_ks_orbit_approx import BinaryKSOrbitApprox
from .trajectories import InertialPair


def temporal_study(root):
    """Exact uniformly moving isolated hole, sampled in a single fixed patch."""
    exact=BinaryKSOrbitApprox((1,0),InertialPair(velocities=((.15,.05,0),(0,0,0))))
    rows=[]
    # Spatial field smoothness is controlled separately using a fixed grid.
    for cadence in (1.,.5,.25):
        p=root/'synthetic'/f'boosted-time-{cadence}'
        if not p.exists():write_dataset(p,np.arange(0,1+cadence/2,cadence),
            [dict(level=0,shape=[13,9,9],origin=[8,-2,-2],spacing=[.5]*3)],
            lambda t,x,l:adm_from_metric(exact,np.r_[t,x]),identifier='exact-boosted-single-hole')
        m=NumericalMetric(p,'cubic');values=[];derivatives=[]
        # Probe spatial nodes: no spatial interpolation error in values or dt.
        for t in np.linspace(.013,.987,47):
            q=np.array([t,10,0,0]);g,dg=m.evaluate(q)
            values.extend((g-exact.inverse(q)).flat)
            derivatives.extend((dg[0]-exact.inverse_derivative(q)[0]).flat)
        rows.append(dict(cadence=cadence,metric_rms=float(np.sqrt(np.mean(np.square(values)))),
                         time_derivative_rms=float(np.sqrt(np.mean(np.square(derivatives))))))
    for a,b in zip(rows,rows[1:]):
        b['metric_order']=float(np.log2(a['metric_rms']/b['metric_rms']))
        b['time_derivative_order']=float(np.log2(a['time_derivative_rms']/b['time_derivative_rms']))
    save(root/'boosted_temporal_convergence.json',rows)


def shadow_study(root):
    """Subpixel classification brackets, not inference from rendered rasters.

    Cell-centered samples avoid the analytical singularity without filling it.
    Explicit absorbers are strictly inside the respective analytical horizons.
    The finite R=14 detector is wholly inside the interpolation domain.
    """
    rows=[]
    for label,exact in [('schwarzschild',Schwarzschild()),('kerr05',Kerr(spin=.5))]:
        c=Camera(position=(12,0,0),target=(0,0,0),up=(0,0,1),time=30,width=96,height=72)
        if label=='schwarzschild':
            expected=np.tan(exact.shadow_angle(12))*96/(2*np.tan(np.pi/6))
        else:expected=reference_radius(shadow_curve(exact,c,2001),c,0.)
        for h in (1.,.5):
            path=root/'synthetic'/f'{label}-shadow-{h}'
            if not path.exists():write_dataset(path,[-100,100],
                [dict(level=0,shape=[round(32/h)]*3,origin=[-16+h/2]*3,spacing=[h]*3,centering='cell')],
                lambda t,x,l:adm_from_metric(exact,np.r_[t,x]),stationary=True,identifier=label+'-shadow-samples')
            m=NumericalMetric(path,'cubic',domain=dict(background='finite_direction_sphere',background_radius=14.,
                capture_spheres=[dict(center=[0,0,0],radius=1.99 if label=='schwarzschild' else 1.7,
                                     description='analytical-control absorber strictly inside known horizon')]))
            settings=Integration(rtol=1e-9,atol=1e-11,max_step=.3,max_affine=150)
            samples=[];lo=expected-1.5;hi=expected+1.5;started=time.perf_counter()
            def classification(radius):
                p=pixel_momentum(m,c,(c.width-1)/2+radius,(c.height-1)/2)
                result=trace(m,c.event,p,settings)
                samples.append(dict(radius_pixels=radius,status=result.status,reason=result.reason,null_error=result.max_null_error))
                return result.status
            valid=classification(lo)=='captured' and classification(hi)=='escaped'
            if valid:
                for _ in range(8):
                    mid=(lo+hi)/2;status=classification(mid)
                    if status=='captured':lo=mid
                    elif status=='escaped':hi=mid
                    else:valid=False;break
            rows.append(dict(model=label,h=h,reference_radius_pixels=expected,valid_bracket=valid,
                bracket_pixels=[lo,hi],midpoint_error_pixels=(lo+hi)/2-expected if valid else None,
                seconds=time.perf_counter()-started,samples=samples))
            print(label,h,'shadow bracket',valid,lo,hi,flush=True)
            save(root/'shadow_boundary_convergence.json',rows)


def throughput(root):
    from .numerical_validation import REAL
    m=NumericalMetric(REAL,'cubic_amr_c1_hierarchy')
    before=time.perf_counter();m.inverse([4,6.5,.23,.17]);load=time.perf_counter()-before
    before=time.perf_counter();n=500
    for i in range(n):m.evaluate([4,6.5+i*1e-5,.23,.17])
    elapsed=time.perf_counter()-before
    save(root/'performance.json',dict(dataset=str(REAL),first_lookup_seconds=load,warm_evaluations=n,
        warm_seconds=elapsed,metric_evaluations_per_second=n/elapsed,active_mapped_bytes=m.data.loaded_bytes,
        serialized_provider_bytes=len(pickle.dumps(m)),cache_budget=m.data.cache_bytes,
        caveat='single measurement; OS page cache and system load affect timing; mmap byte count is not peak RSS'))


def amr_study(root):
    from .numerical_fixtures import nested_boxes
    from .geodesic import TraceDomain
    rows=[]
    for label,exact in [('schwarzschild',Schwarzschild()),('kerr05',Kerr(spin=.5))]:
        c=Camera(position=(6.3,.17,.23),target=(7.3,.17,.23),up=(0,0,1),time=4)
        settings=Integration(rtol=1e-10,atol=1e-12,max_step=.1,max_affine=4)
        domain=TraceDomain(lambda x:1.,lambda x:-1.,lambda z:np.array([1.,0,0]))
        _,ref=trace(exact,c.event,c.momentum(exact,[1,.05,.02]),settings,domain=domain,return_solution=True)
        for h in (1.,.5):
            p=root/'synthetic'/f'{label}-amr-{h}'
            if not p.exists():write_dataset(p,[-30,30],nested_boxes(2,spacing=h),
                lambda t,x,l:adm_from_metric(exact,np.r_[t,x]),stationary=True,identifier=label+'-nested-samples')
            m=NumericalMetric(p,'cubic_amr_c1_hierarchy')
            result,sol=trace(m,c.event,c.momentum(m,[1,.05,.02]),settings,return_solution=True)
            times=np.linspace(0,min(ref.t[-1],sol.t[-1]),101);error=np.max(abs(sol.sol(times)-ref.sol(times)),axis=1)
            rows.append(dict(model=label,h=h,event_error=float(max(error[:4])),momentum_error=float(max(error[4:])),
                selected_levels=result.numerical_diagnostics['selected_levels'],null_error=result.max_null_error,
                status=result.status,reason=result.reason))
    save(root/'analytic_amr_trajectories.json',rows)


def domain_study(root):
    from .numerical_validation import REAL
    rows=[];c=Camera(position=(6.5,.23,.17),target=(7.5,.23,.17),up=(0,0,1),time=4)
    for radius in (7.5,8.,8.5,9.):
        for tolerance in (1e-8,1e-10,1e-12):
            m=NumericalMetric(REAL,'cubic_amr_c1_hierarchy',domain=dict(background='finite_direction_sphere',background_radius=radius))
            r,sol=trace(m,c.event,c.momentum(m,[1,.1,.05]),Integration(rtol=tolerance,atol=tolerance/100,max_step=.08,max_affine=8),return_solution=True)
            rows.append(dict(radius=radius,rtol=tolerance,status=r.status,reason=r.reason,null_error=r.max_null_error,
                direction=None if r.direction is None else r.direction.tolist(),endpoint=sol.y[:4,-1].tolist()))
    def separation(a,b):return float(np.arctan2(np.linalg.norm(np.cross(a,b)),np.dot(a,b)))
    for row in rows:
        ref=next(r for r in rows if r['radius']==row['radius'] and r['rtol']==1e-12)
        if row['direction'] is not None and ref['direction'] is not None:row['tolerance_direction_error_radians']=separation(row['direction'],ref['direction'])
    tight=[r for r in rows if r['rtol']==1e-12]
    for a,b in zip(tight,tight[1:]):
        if a['direction'] is not None and b['direction'] is not None:b['direction_change_from_previous_radius']=separation(a['direction'],b['direction'])
    save(root/'real_domain_convergence.json',rows)


def parallel_study(root):
    from .numerical_validation import REAL
    from .render import render
    from .background import CelestialGrid
    rows=[];c=Camera(position=(6.5,.23,.17),target=(7.5,.23,.17),up=(0,0,1),time=4,width=4,height=3,fov_degrees=40)
    for workers in (1,2):
        m=NumericalMetric(REAL,'cubic_amr_c1_hierarchy',domain=dict(background='finite_direction_sphere',background_radius=8.))
        start=time.perf_counter();report=render(m,c,Integration(rtol=1e-10,atol=1e-12,max_step=.06,max_affine=6),root/f'parallel-{workers}',background=CelestialGrid(),workers=workers)
        rows.append(dict(workers=workers,wall_seconds=time.perf_counter()-start,render_seconds=report['elapsed_seconds'],counts=report['counts'],rays=12))
    with np.load(root/'parallel-1/rays.npz') as a,np.load(root/'parallel-2/rays.npz') as b:
        for key in a.files:np.testing.assert_array_equal(a[key],b[key])
    save(root/'parallel_performance.json',dict(measurements=rows,all_saved_arrays_exactly_equal=True,
        speedup=rows[0]['wall_seconds']/rows[1]['wall_seconds'],caveat='12 rays; startup and validation overhead significant; not a scaling benchmark'))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--shadow',action='store_true');p.add_argument('--temporal',action='store_true');p.add_argument('--performance',action='store_true')
    p.add_argument('--amr',action='store_true')
    p.add_argument('--domain',action='store_true');p.add_argument('--parallel',action='store_true')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    if a.temporal:temporal_study(a.output)
    if a.shadow:shadow_study(a.output)
    if a.performance:throughput(a.output)
    if a.amr:amr_study(a.output)
    if a.domain:domain_study(a.output)
    if a.parallel:parallel_study(a.output)


if __name__=='__main__':main()
