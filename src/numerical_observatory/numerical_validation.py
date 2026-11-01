"""Controlled GMDS/C++ parity, analytic convergence and small real-data preview."""
import argparse,json,time,pickle
from dataclasses import asdict,replace
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from .numerical_metric import NumericalMetric
from .numerical_fixtures import write_dataset,manufactured_fields,nested_boxes,adm_from_metric
from .cpp_oracle import CppOracle
from .metric import Schwarzschild,ETA
from .kerr import Kerr
from .boosted_ks import lorentz_boost
from .binary_ks_orbit_approx import BinaryKSOrbitApprox
from .trajectories import InertialPair
from .camera import Camera
from .geodesic import trace,Integration
from .background import Starfield,CelestialGrid
from .diagnostic_render import shade,save_image
from .render import render

ROOT=Path(__file__).resolve().parents[2]
REAL=ROOT/'reference/geodesics-cpp/results/amr-composite-multilevel/carpetx-native-gmds'


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def fixture(output,levels):
    p=Path(output)/f'manufactured-{levels}'
    if not p.exists():write_dataset(p,[-20,0,20],nested_boxes(levels),manufactured_fields)
    return p


def parity(output,oracle,backend='python'):
    output=Path(output);records=[];golden={}
    for mode,levels in [('linear',3),('cubic',3),('cubic_amr_c1',2),('cubic_amr_c1_box',2),('cubic_amr_c1_hierarchy',3)]:
        p=fixture(output/'fixtures',levels);options={}
        auto=NumericalMetric(p,'cubic_amr_c1_hierarchy')
        if mode=='cubic_amr_c1_box':options['box_transition']=asdict(auto.bands[0])
        if mode=='cubic_amr_c1':
            b=auto.bands[0];options['transition']=dict(axis=0,coarse_level=0,fine_level=1,lower=b.inner_upper[0],upper=b.outer_upper[0],fine_on_lower=True)
        m=NumericalMetric(p,mode,backend=backend,**options)
        xs=[4.,6.,6.5,7.,7.5,8.,8.5,9.,10.,11.,11.5,12.,12.5,13.,14.,15.]
        events=[np.array([.37,x,.173,.219]) for x in xs]
        # Every face/edge/corner branch, including exact/one-ULP endpoints.
        if mode=='cubic_amr_c1_hierarchy':
            for b in m.bands:
                for d in range(3):
                    for edge in (b.outer_lower[d],b.inner_lower[d],b.inner_upper[d],b.outer_upper[d]):
                        for delta in (-np.inf,None,np.inf):
                            x=np.array([10.,0,0]);x[d]=edge if delta is None else np.nextafter(edge,delta)
                            events.append(np.r_[.37,x])
                for lower in (True,False):
                    endpoint=np.array(b.outer_lower if lower else b.outer_upper);events.append(np.r_[.37,endpoint])
                    endpoint=np.array(b.inner_lower if lower else b.inner_upper);events.append(np.r_[.37,endpoint])
        values,selected=oracle.probes(p,mode,events);maximum=0.;matches=0;valid=0;actual=[]
        for event,expected,ids in zip(events,values,selected):
            try:g,dg=m.evaluate(event);value=np.concatenate((g[None],dg));actual.append(value)
            except ValueError:
                actual.append(np.full((5,4,4),np.nan))
                if expected is not None:raise RuntimeError(f'Python rejected C++-valid {mode} at {event}')
                matches+=1;continue
            if expected is None:raise RuntimeError(f'Python accepted C++-invalid {mode} at {event}')
            error=float(np.max(abs(value-expected)));maximum=max(maximum,error);valid+=1;matches+=1
            if ids and set(ids)!=m.last_levels:raise RuntimeError('C++/Python required-level mismatch')
        if maximum>2e-12:raise RuntimeError('metric/derivative parity exceeded roundoff target')
        row=dict(mode=mode,events=len(events),valid=valid,classification_matches=matches,max_component_difference=maximum,options=options)
        records.append(row);golden[mode+'_events']=np.array(events);golden[mode+'_values']=np.array([v if v is not None else np.full((5,4,4),np.nan) for v in values])
    np.savez_compressed(output/'synthetic_cpp_golden.npz',**golden)
    # Independent raw file reading and common Eulerian frame/momentum.
    p=fixture(output/'fixtures',3);m=NumericalMetric(p,'cubic_amr_c1_hierarchy',backend=backend)
    patch=m.data.snapshots[0][0];nodes=[0,17,129,1000]
    raw=oracle.run(p/patch.file,'linear','raw',[[i] for i in nodes])
    expected=np.array([list(map(float,line.split())) for line in raw]);actual=np.memmap(p/patch.file,dtype='<f8',mode='r').reshape(-1,10)[nodes]
    np.testing.assert_array_equal(expected,actual)
    event=np.array([1.5,6.3,.17,.23]);direction=np.array([1.,.1,.1])
    reply=oracle.run(p,'cubic_amr_c1_hierarchy','eulerian',[np.r_[event,direction]])[0]
    values=np.array(list(map(float,reply.split()[1:])));frame=values[:16].reshape(4,4)
    c=Camera(position=tuple(event[1:]),time=event[0],target=(7.3,.17,.23),up=(0,0,1),observer_frame=tuple(map(tuple,frame)))
    frame_error=float(np.max(abs(frame@m.covariant(event)@frame.T-ETA)))
    p0=c.momentum(m,direction);momentum_error=float(np.max(abs(p0-values[16:20])))
    # Shared explicit state, including past-directed sign; no frame-default mismatch.
    comparisons=[]
    for label,path,provider,camera,ray,background in [('manufactured',p,m,c,p0,0.)]:
        s=Integration(rtol=1e-11,atol=1e-13,max_step=.08,max_affine=4)
        a,sol=trace(provider,camera.event,ray,s,return_solution=True)
        lines=oracle.run(path,provider.mode,'trace',[np.r_[camera.event,ray,s.max_affine,s.max_step,1e-12,1e-14,background,0.]])
        reference=np.array([list(map(float,line.split())) for line in lines[1:]])
        inside=reference[:,0]<=sol.t[-1];reference=reference[inside]
        error=np.max(abs(sol.sol(reference[:,0]).T-reference[:,1:]),axis=0)
        np.savez_compressed(output/(label+'_trajectories.npz'),cpp=reference,python=sol.sol(reference[:,0]).T)
        comparisons.append(dict(label=label,python_status=a.status,reason=a.reason,cpp_status=lines[0],
            max_event_error=float(max(error[:4])),max_momentum_error=float(max(error[4:])),null_error=a.max_null_error,selected_levels=a.numerical_diagnostics['selected_levels']))
    result=dict(probes=records,raw_nodes_bitwise_equal=True,frame_orthonormality_error=frame_error,initial_momentum_error=momentum_error,
                trajectories=comparisons,oracle=oracle.config)
    if frame_error>1e-10 or momentum_error>2e-12 or max(r['max_event_error'] for r in comparisons)>2e-8:
        raise RuntimeError('cross-language state/frame comparison failed')
    save(output/'cpp_parity.json',result)
    return result


def real_parity(output,oracle,dataset=REAL,backend='python'):
    output=Path(output);m=NumericalMetric(dataset,'cubic_amr_c1_hierarchy',backend=backend)
    events=[np.array([4.,x,.23,.17]) for x in (2.,3.,3.25,3.5,4.,5.,6.,6.5,7.,8.,10.,12.)]
    cpp,ids=oracle.probes(dataset,m.mode,events);errors=[]
    for x,expected,levels in zip(events,cpp,ids):
        g,dg=m.evaluate(x);errors.append(float(np.max(abs(np.concatenate((g[None],dg))-expected))))
        if set(levels)!=m.last_levels:raise RuntimeError('real required-level mismatch')
    c=Camera(position=(6.5,.23,.17),time=4,target=(7.5,.23,.17),up=(0,0,1));p=c.momentum(m,[1,.1,.05])
    m.domain_options=dict(background='finite_direction_sphere',background_radius=8.)
    s=Integration(rtol=1e-11,atol=1e-13,max_step=.04,max_affine=5)
    r,sol=trace(m,c.event,p,s,return_solution=True)
    lines=oracle.run(dataset,m.mode,'trace',[np.r_[c.event,p,5,.04,1e-12,1e-14,8,0]])
    reference=np.array([list(map(float,line.split())) for line in lines[1:]])
    safe=reference[:,0]<=sol.t[-1];error=np.max(abs(sol.sol(reference[safe,0]).T-reference[safe,1:]),axis=0)
    # Endpoint is independently located by DP5 reintegration vs DOP853 dense root.
    endpoint_error=float(np.max(abs(sol.y[:,-1]-reference[-1,1:])))
    last=reference[-1,1:];tangent=(m.inverse(last[:4])@last[4:])[1:];tangent/=np.linalg.norm(tangent)
    direction_error=float(np.arctan2(np.linalg.norm(np.cross(r.direction,tangent)),r.direction@tangent))
    result=dict(dataset=m.description(),events=len(events),max_metric_derivative_difference=max(errors),
                python_status=r.status,cpp_status=lines[0],max_event_error=float(max(error[:4])),max_momentum_error=float(max(error[4:])),
                endpoint_error=endpoint_error,source_direction_error=direction_error,null_error=r.max_null_error,trace_diagnostics=r.numerical_diagnostics)
    np.savez_compressed(output/'real_cpp_trajectories.npz',cpp=reference,python=sol.sol(reference[safe,0]).T)
    save(output/'real_cpp_parity.json',result)
    if max(errors)>2e-12 or endpoint_error>2e-8 or direction_error>2e-9 or r.status!='escaped':raise RuntimeError('real CarpetX parity discrepancy')
    return result


def analytic_studies(output):
    output=Path(output);rows=[]
    for label,exact in [('schwarzschild',Schwarzschild()),('kerr_spin05',Kerr(spin=.5))]:
        for h in (1.,.5,.25):
            path=output/'synthetic'/f'{label}-{h}'
            if not path.exists():
                nx=round(8/h)+1;ny=round(6/h)+1
                write_dataset(path,[-20,20],[dict(level=0,shape=[nx,ny,ny],origin=[6,-3,-3],spacing=[h]*3)],
                              lambda t,x,l:adm_from_metric(exact,np.r_[t,x]),stationary=True,identifier='analytic-'+label)
            m=NumericalMetric(path,'cubic');ve=[];de=[]
            for i in range(24):
                q=np.array([1.,8+4*(i+.173)/24,.7*np.sin(i+1),.6*np.cos(i+1)])
                g,dg=m.evaluate(q);ve.extend((g-exact.inverse(q)).flat);de.extend((dg[1:]-exact.inverse_derivative(q)[1:]).flat)
            c=Camera(position=(8,.17,.23),target=(9,.17,.23),up=(0,0,1),time=2);p=c.momentum(exact,[1,.1,.1])
            s=Integration(rtol=1e-11,atol=1e-13,max_step=.05,max_affine=2)
            numerical_p=c.momentum(m,[1,.1,.1])
            r,sol=trace(m,c.event,numerical_p,s,return_solution=True)
            # Exact provider under a matching finite, no-physical-event domain.
            from .geodesic import TraceDomain
            domain=TraceDomain(lambda x:1.,lambda x:-1.,lambda z:np.array([1.,0,0]))
            er,es=trace(exact,c.event,p,s,domain=domain,return_solution=True)
            ts=np.linspace(0,min(sol.t[-1],es.t[-1]),101);delta=np.max(abs(sol.sol(ts)-es.sol(ts)),axis=1)
            rows.append(dict(model=label,h=h,metric_rms=float(np.sqrt(np.mean(np.array(ve)**2))),derivative_rms=float(np.sqrt(np.mean(np.array(de)**2))),
                             max_event_error=float(max(delta[:4])),max_momentum_error=float(max(delta[4:])),initial_momentum_error=float(max(abs(p-numerical_p))),status=r.status,reason=r.reason,null_error=r.max_null_error))
    for label in set(r['model'] for r in rows):
        group=[r for r in rows if r['model']==label]
        for a,b in zip(group,group[1:]):
            b['metric_observed_order']=float(np.log2(a['metric_rms']/b['metric_rms']));b['derivative_observed_order']=float(np.log2(a['derivative_rms']/b['derivative_rms']))
    save(output/'analytic_convergence.json',rows)
    return rows


def demo(output,dataset=REAL,workers=1):
    output=Path(output);m=NumericalMetric(dataset,'cubic_amr_c1_hierarchy',domain=dict(background='finite_direction_sphere',background_radius=8.))
    c=Camera(position=(6.5,.23,.17),target=(7.5,.23,.17),up=(0,0,1),time=4,fov_degrees=40,width=24,height=18)
    s=Integration(rtol=1e-10,atol=1e-12,max_step=.06,max_affine=6)
    print('START real numerical preview 24x18',flush=True)
    report=render(m,c,s,output/'real_preview',background=Starfield(),workers=workers)
    with np.load(output/'real_preview/rays.npz') as z:
        save_image(output/'real_preview/grid.png',shade(z['directions'],z['status'],CelestialGrid(),1))
    atlas=Image.new('RGB',(768,314),(246,247,249));draw=ImageDraw.Draw(atlas)
    for i,(file,title) in enumerate([('shadow.png','Imported CarpetX: finite starfield boundary'),('grid.png','Same numerical rays: celestial grid')]):
        with Image.open(output/'real_preview'/file) as im:atlas.paste(im.resize((384,288),Image.Resampling.NEAREST),(i*384,26))
        draw.text((i*384+6,7),title,fill=(15,20,30))
    atlas.save(output/'real_preview/comparison.png');print('DONE numerical preview',report['counts'],flush=True)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('output/numerical-validation'))
    parser.add_argument('--oracle',type=Path)
    parser.add_argument('--backend',choices=('python','numpy','numba'),default='python',help='backend for independent C++ parity probes and trajectories')
    parser.add_argument('--analytic',action='store_true');parser.add_argument('--demo',action='store_true')
    parser.add_argument('--workers',type=int,default=1);parser.add_argument('--real-dataset',type=Path,default=REAL)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    if args.oracle:
        oracle=CppOracle(args.oracle);parity(args.output,oracle,args.backend);real_parity(args.output,oracle,args.real_dataset,args.backend)
    if args.analytic:analytic_studies(args.output)
    if args.demo:demo(args.output,args.real_dataset,args.workers)


if __name__=='__main__':main()
