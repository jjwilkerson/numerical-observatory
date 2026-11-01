"""Small reproducible orbiting-binary audits and three 96x72 control previews."""
import argparse
from dataclasses import replace,asdict
import hashlib,json,time
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from .binary_ks_orbit_approx import BinaryKSOrbitApprox
from .binary_ks_approx import BinaryKSApprox
from .trajectories import CircularOrbit,InertialPair
from .boosted_ks import lorentz_boost
from .curvature_diagnostic import ricci
from .cli import load_configuration
from .camera import Camera
from .metric import ETA,Schwarzschild
from .geodesic import trace,hamilton_rhs,TraceDomain
from .background import Starfield,CelestialGrid
from .render import render
from .diagnostic_render import shade,save_image


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def angle(a,b):return float(np.arctan2(np.linalg.norm(np.cross(a,b)),np.dot(a,b)))


def inertial_ray_audit(settings,output):
    velocity=(.15,-.07,0);L=lorentz_boost(velocity);inv=np.linalg.inv(L)
    metric=BinaryKSOrbitApprox((1,0),InertialPair(velocities=(velocity,(0,0,0))))
    ref=Schwarzschild();camera=Camera(position=(0,0,20));X=camera.event;x=inv@X
    settings=replace(settings,rtol=1e-11,atol=1e-13,escape_radius=150,max_affine=500,max_step=20)
    def sky(z):
        v=inv@(ref.inverse(z[:4])@z[4:]);return v[1:]/np.linalg.norm(v[1:])
    domain=TraceDomain(lambda e:np.linalg.norm(e[1:])-1.99,
                       lambda e:np.linalg.norm((inv@e)[1:])-settings.escape_radius,sky)
    arrays={};rows=[]
    for i,direction in enumerate(([0,0,-1],[.4,0,-1],[0,.7,-1])):
        p=camera.momentum(ref,direction);gp=L.T@p
        a,sa=trace(metric,x,gp,settings,return_solution=True)
        b,sb=trace(ref,X,p,settings,domain=domain,return_solution=True)
        t=np.linspace(0,min(sa.t[-1],sb.t[-1]),101);reference=sb.sol(t)
        expected=np.vstack((inv@reference[:4],L.T@reference[4:]));actual=sa.sol(t)
        arrays.update({f'global_{i}':actual,f'transformed_reference_{i}':expected,f'affine_{i}':t})
        row=dict(local_camera_direction=direction,global_event=x.tolist(),global_momentum=gp.tolist(),
                 status=a.status,reference_status=b.status,max_event_error=float(np.max(abs(actual[:4]-expected[:4]))),
                 max_covector_error=float(np.max(abs(actual[4:]-expected[4:]))))
        if a.direction is not None:row['source_direction_error']=angle(a.direction,b.direction)
        rows.append(row)
    np.savez_compressed(Path(output)/'inertial_trajectories.npz',**arrays)
    if any(r['status']!=r['reference_status'] or r['max_event_error']>3e-7 or r['max_covector_error']>3e-7 for r in rows):
        raise RuntimeError('independent inertial ray comparison failed')
    return dict(velocity=velocity,settings=asdict(settings),comparison='same affine normalization, transformed observer/p, matched GLOBAL finite escape sphere',rays=rows)


def unequal_mass_audit(settings,output):
    masses=(.7,.3);metric=BinaryKSOrbitApprox(masses,CircularOrbit(masses,30))
    camera=Camera(position=(0,0,75));rows=[]
    settings=replace(settings,rtol=1e-10,atol=1e-12,escape_radius=500,max_affine=1500,max_step=30)
    for direction in ([.11,-.052,-1],[-.25,.12,-1],[.4,.15,-1]):
        r=trace(metric,camera.event,camera.momentum(metric,direction),settings)
        rows.append(dict(camera_direction=direction,status=r.status,reason=r.reason,null_error=r.max_null_error,diagnostics=r.binary_diagnostics))
    if [r['status'] for r in rows]!=['captured','captured','escaped']:
        raise RuntimeError('unequal-mass reference classifications disagree')
    save(Path(output)/'unequal_mass_control.json',dict(spacetime=metric.description(),camera=asdict(camera),integration=asdict(settings),rays=rows))


def audit(m,c,s,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    result=dict(metric=m.description(),camera=asdict(c),integration=asdict(s))
    derivative_rows=[]
    for xyz in ([15,1.5,0],[-10,4,2],[2,3,80]):
        x=np.r_[27.,xyz];g=m.covariant(x);dg=m.covariant_derivative(x)
        for a in range(4):
            unit=np.eye(4)[a];cs=np.imag(m.inverse(x+1j*1e-20*unit))/1e-20
            errors=[]
            for h in (.01,.001,.0001):
                fd=(-m.inverse(x+2*h*unit)+8*m.inverse(x+h*unit)-8*m.inverse(x-h*unit)+m.inverse(x-2*h*unit))/(12*h)
                errors.append(float(np.max(abs(fd-cs))))
            derivative_rows.append(dict(event=x.tolist(),axis=a,complex_step_error=float(np.max(abs(m.inverse_derivative(x)[a]-cs))),
                                        fd_steps=[.01,.001,.0001],fd_errors=errors))
    result['derivatives']=derivative_rows
    controls=[]
    for velocity in ((0,0,0),(.15,-.08,.1),(-.2,0,0)):
        moving=BinaryKSOrbitApprox((1,0),InertialPair(velocities=(velocity,(0,0,0))))
        L=lorentz_boost(velocity);inv=np.linalg.inv(L);x=np.array([7.,4,3,2]);X=L@x;ref=Schwarzschild()
        expected=np.einsum('ba,mi,nj,bij->amn',L,inv,inv,ref.inverse_derivative(X))
        controls.append(dict(velocity=velocity,covariant_error=float(np.max(abs(moving.covariant(x)-L.T@ref.covariant(X)@L))),
            inverse_error=float(np.max(abs(moving.inverse(x)-inv@ref.inverse(X)@inv.T))),
            inverse_derivative_error=float(np.max(abs(moving.inverse_derivative(x)-expected)))))
    result['inertial_tensor_controls']=controls
    result['inertial_ray_control']=inertial_ray_audit(s,output)
    unequal_mass_audit(s,output)
    stationary=BinaryKSApprox(m.masses,tuple(map(tuple,m.trajectory.state(m.trajectory.epoch)[0])))
    rows=[]
    for rate in (.1,.01,.001,0):
        control=BinaryKSOrbitApprox(m.masses,replace(m.trajectory,rate_scale=rate))
        errors=[float(np.max(abs(control.covariant(np.r_[t,9.,2,5])-stationary.covariant(np.r_[t,9.,2,5])))) for t in (-70,0,100)]
        rows.append(dict(rate_scale=rate,times=[-70,0,100],covariant_errors=errors))
    result['zero_rate_family']=rows
    # Locate a representative boundary in a documented camera-direction slice.
    left,right=.22,.30
    for _ in range(9):
        mid=(left+right)/2;r=trace(m,c.event,c.momentum(m,[mid,-.09,-1]),s)
        if r.status=='captured':left=mid
        elif r.status=='escaped':right=mid
        else:raise RuntimeError('unresolved boundary search: '+r.reason)
    result['boundary_slice']=dict(y_direction=-.09,captured_x=left,escaped_x=right,width=right-left)
    directions=([.4,.2,-1],[.26,-.09,-1],[.18,-.09,-1],[-.18,.09,-1],[0,0,-1],[right+.0002,-.09,-1])
    ray_rows=[]
    for direction in directions:
        p=c.momentum(m,direction);rows=[]
        for tol in (1e-7,1e-9,1e-11):
            r,sol=trace(m,c.event,p,replace(s,rtol=tol,atol=tol/100),return_solution=True)
            row=dict(rtol=tol,status=r.status,reason=r.reason,null_error=r.max_null_error,
                     diagnostics=r.binary_diagnostics,end_event=None if sol is None else sol.y[:4,-1].tolist(),
                     direction=None if r.direction is None else r.direction.tolist())
            if sol is not None:
                z=sol.sol(sol.t[-1]/3);h=.001
                row['pt_equation_error']=float(abs((sol.sol(sol.t[-1]/3+h)[4]-sol.sol(sol.t[-1]/3-h)[4])/(2*h)-hamilton_rhs(m,z)[4]))
                row['past_time_monotonic']=bool(np.all(np.diff(sol.y[0])<0))
            rows.append(row)
        for row in rows[:-1]:
            row['event_difference_from_tight']=float(np.max(abs(np.array(row['end_event'])-rows[-1]['end_event'])))
            if row['direction'] is not None and rows[-1]['direction'] is not None:row['source_difference_from_tight']=angle(row['direction'],rows[-1]['direction'])
        ray_rows.append(dict(camera_direction=direction,runs=rows))
    result['ray_convergence']=ray_rows
    source=[]
    for direction in (directions[0],directions[1],directions[-1]):
        rows=[];p=c.momentum(m,direction)
        for radius in (250,500,1000,2000):
            r=trace(m,c.event,p,replace(s,rtol=1e-11,atol=1e-13,escape_radius=radius,max_affine=5000,max_step=40))
            rows.append(dict(radius=radius,status=r.status,direction=None if r.direction is None else r.direction.tolist(),null_error=r.max_null_error))
        source.append(dict(camera_direction=direction,runs=rows,successive_direction_changes=[angle(a['direction'],b['direction']) for a,b in zip(rows,rows[1:])]))
    result['source_radius']=source
    sensitivities=[]
    for direction in (directions[2],directions[-1]):
        rows=[];p=c.momentum(m,direction)
        for fraction in (.98,.995,.999):
            r,sol=trace(m,c.event,p,replace(s,binary_capture_fraction=fraction),return_solution=True)
            rows.append(dict(fraction=fraction,status=r.status,reason=r.reason,end_event=sol.y[:4,-1].tolist(),
                             direction=None if r.direction is None else r.direction.tolist()))
        sensitivities.append(dict(camera_direction=direction,runs=rows))
    result['capture_sensitivity']=sensitivities
    # Curvature numerical error is calibrated on an EXACT boosted vacuum hole.
    exact=BinaryKSOrbitApprox((1,0),InertialPair(velocities=((.15,.03,0),(0,0,0))))
    curvature=[]
    for label,metric,event in [('exact_inertial',exact,[0,3,1,2]),('orbit_near',m,[0,15,2,1]),
                                ('orbit_intermediate',m,[0,4,3,7]),('orbit_far',m,[0,2,3,80])]:
        rows=[]
        for h in (.03,.01,.003):rows.append(dict(step=h,**ricci(metric,event,h)))
        curvature.append(dict(label=label,event=event,values=rows))
    result['curvature']=curvature
    result['passed']=(max(x['complex_step_error'] for x in derivative_rows)<1e-12
        and max(x['inverse_derivative_error'] for x in controls)<1e-12
        and all(len(set(r['status'] for r in ray['runs']))==1 and ray['runs'][-1]['status']!='unresolved' for ray in ray_rows)
        and all(ray['runs'][1].get('source_difference_from_tight',0)<2e-6 for ray in ray_rows)
        and all(ray['runs'][1]['diagnostics']['helical_error']<s.null_tolerance for ray in ray_rows)
        and all(all(a['status']=='escaped' for a in row['runs']) and row['successive_direction_changes'][-1]<row['successive_direction_changes'][0] for row in source)
        and all(len(set(a['status'] for a in row['runs']))==1 for row in sensitivities))
    save(output/'validation.json',result)
    if not result['passed']:raise RuntimeError('orbit metric audit failed; inspect validation.json')
    return result


def previews(m,c,s,output,workers):
    output=Path(output);records=[]
    frozen=BinaryKSApprox(m.masses,tuple(map(tuple,m.trajectory.state(m.trajectory.epoch)[0])))
    cases=[('phase0',m,c),('phase45',m,replace(c,time=c.time+np.pi/4/m.trajectory.helical_omega)),
           ('stationary_zero_rate_control',frozen,c)]
    for name,metric,camera in cases:
        print('START preview',name,flush=True);start=time.perf_counter()
        last=[0.]
        def progress(done,total):
            if time.perf_counter()-last[0]>10 or done==total:
                print(name,'rows',done,'/',total,'elapsed',round(time.perf_counter()-start,1),flush=True);last[0]=time.perf_counter()
        report=render(metric,camera,s,output/name,background=Starfield(),workers=workers,progress=progress)
        with np.load(output/name/'rays.npz') as z:
            save_image(output/name/'grid.png',shade(z['directions'],z['status'],CelestialGrid(),1))
        save(output/name/'grid_metadata.json',dict(source_rays='rays.npz',additional_integrations=0,
             background=CelestialGrid().description,camera=report['camera'],spacetime=report['spacetime']))
        records.append(dict(name=name,camera=report['camera'],spacetime=report['spacetime'],counts=report['counts'],
                            maxima=report['maxima'],runtime=time.perf_counter()-start,grid_additional_integrations=0))
        save(output/'preview_records.json',records)
        print('DONE preview',name,records[-1]['runtime'],report['counts'],flush=True)
    figures(output)


def figures(output):
    output=Path(output);records=json.loads((output/'preview_records.json').read_text())
    atlas=Image.new('RGB',(768,3*314),(246,247,249));draw=ImageDraw.Draw(atlas)
    for i,row in enumerate(records):
        for j,file in enumerate(('shadow.png','grid.png')):
            with Image.open(output/row['name']/file) as im:atlas.paste(im.convert('RGB').resize((384,288),Image.Resampling.NEAREST),(j*384,i*314+26))
            draw.text((j*384+8,i*314+7),row['name']+(' / starfield' if j==0 else ' / same-ray grid'),fill=(15,20,30))
    atlas.save(output/'comparison.png')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path('examples/binary_ks_orbit_approx/circular_30M.toml'))
    parser.add_argument('--output',type=Path,default=Path('output/orbit-binary-experiment'))
    parser.add_argument('--render',action='store_true')
    parser.add_argument('--figures-only',action='store_true')
    parser.add_argument('--workers',type=int,default=1)
    args=parser.parse_args()
    if args.figures_only:figures(args.output);return
    m,c,s=load_configuration(args.config)
    audit(m,c,s,args.output)
    if args.render:previews(m,c,s,args.output,args.workers)


if __name__=='__main__':main()
