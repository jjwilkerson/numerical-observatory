"""Reproducible small audits and a shared-ray binary KS preview pair."""
import argparse
from dataclasses import replace,asdict
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from .binary_ks_approx import BinaryKSApprox
from .binary_support import BinaryDomain
from .metric import Schwarzschild,ETA
from .camera import Camera
from .geodesic import Integration,trace
from .cli import load_configuration
from .background import Starfield,CelestialGrid
from .render import render
from .diagnostic_render import shade,save_image


def angle(a,b):
    return float(np.arctan2(np.linalg.norm(np.cross(a,b)),np.dot(a,b)))


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def audit(metric,camera,settings,output):
    if type(metric) is not BinaryKSApprox:raise ValueError('binary validation requires binary_ks_approx')
    policy=BinaryDomain(metric,settings)
    inversion=[];derivative=[];determinants=[];minimum_signature=1.;signature_failures=0;n=0
    # A sampled exterior volume, not a proof of globally valid binary geometry.
    for x in np.linspace(-15,15,11)*metric.mass:
        for y in np.linspace(-8,8,9)*metric.mass:
            for z in np.linspace(-10,10,9)*metric.mass:
                event=np.array([0.,x,y,z])
                if policy.capture(event)<=0:continue
                margin=metric.signature_margin(event)
                minimum_signature=min(minimum_signature,margin);n+=1
                g=metric.covariant(event)
                signature_failures+=int(np.count_nonzero(np.linalg.eigvalsh(g)<0)!=1)
                inversion.append(float(np.max(abs(g@metric.inverse(event)-np.eye(4)))))
                determinants.append(abs(np.linalg.det(g)+margin))
    for xyz in ([2,3,5],[-6,2,0],[6,0,1.5],[0,0,30]):
        event=np.r_[7.,np.asarray(xyz)*metric.mass]
        for a in range(4):
            h=1e-5*metric.mass;dx=np.eye(4)[a]*h
            derivative.append(float(np.max(abs(metric.inverse_derivative(event)[a]-(metric.inverse(event+dx)-metric.inverse(event-dx))/(2*h)))))
    frame=camera.tetrad(metric)
    output=Path(output)
    result=dict(metric=metric.description(),camera=asdict(camera),integration=asdict(settings),
                exterior_grid_samples=n,exterior_grid_signature_failures=signature_failures,
                exterior_grid_minimum_signature_margin=minimum_signature,
                max_inverse_identity_error=max(inversion),max_determinant_identity_error=max(determinants),
                max_inverse_derivative_finite_difference_error=max(derivative),
                tetrad_error=float(np.max(abs(frame@metric.covariant(camera.event)@frame.T-ETA))))
    convergences=[]
    directions=([.14,.07,-1],[-.14,.07,-1],[.25,.15,-1],[0,.2,-1],[-.133,0,-1],[.133,0,-1],[0,0,-1])
    for direction in directions:
        p=camera.momentum(metric,direction);rows=[]
        for tol in (1e-7,1e-9,1e-11):
            s=replace(settings,rtol=tol,atol=tol/100)
            r,sol=trace(metric,camera.event,p,s,return_solution=True)
            rows.append(dict(rtol=tol,status=r.status,reason=r.reason,null_error=r.max_null_error,
                             energy_drift=r.energy_drift,end_event=None if sol is None else sol.y[:4,-1].tolist(),
                             source_direction=None if r.direction is None else r.direction.tolist(),
                             minimum_signature_margin=r.binary_diagnostics['minimum_signature_margin']))
        for row in rows[:-1]:
            row['endpoint_component_difference_from_tight']=float(np.max(abs(np.array(row['end_event'])-rows[-1]['end_event'])))
            if row['source_direction'] is not None and rows[-1]['source_direction'] is not None:
                row['source_angle_from_tight']=angle(row['source_direction'],rows[-1]['source_direction'])
        convergences.append(dict(camera_direction=direction,runs=rows))
    result['ray_convergence']=convergences
    radius_rows=[]
    for direction in directions[:4]:
        p=camera.momentum(metric,direction);rows=[]
        for radius in (250,500,1000,2000):
            s=replace(settings,rtol=1e-11,atol=1e-13,escape_radius=radius*metric.mass,
                      max_affine=5000*metric.mass,max_step=40*metric.mass)
            r=trace(metric,camera.event,p,s)
            rows.append(dict(radius=radius,status=r.status,direction=r.direction.tolist() if r.direction is not None else None))
        differences=[angle(a['direction'],b['direction']) for a,b in zip(rows,rows[1:])]
        radius_rows.append(dict(camera_direction=direction,runs=rows,successive_direction_changes_radians=differences))
    result['escape_convergence']=radius_rows
    # Independent existing Schwarzschild provider, at the SAME finite cutoff.
    one=BinaryKSApprox((1.,0.),((0,0,0),(6,0,0)))
    single_camera=Camera(position=(0,0,20));single=[]
    for direction in ([0,0,-1],[.4,0,-1],[0,.7,-1]):
        p=single_camera.momentum(one,direction)
        s=Integration(rtol=1e-11,atol=1e-13,escape_radius=1000,max_affine=2500,max_step=20)
        a,sa=trace(one,single_camera.event,p,s,return_solution=True)
        b,sb=trace(Schwarzschild(),single_camera.event,p,s,return_solution=True)
        sample=np.linspace(0,min(sa.t[-1],sb.t[-1]),101)
        diff=float(np.max(abs(sa.sol(sample)-sb.sol(sample))))
        row=dict(camera_direction=direction,binary_status=a.status,schwarzschild_status=b.status,
                 max_state_component_difference=diff)
        if a.direction is not None:
            vel=Schwarzschild().inverse(sb.y[:4,-1])@sb.y[4:,-1]
            row['matched_finite_source_direction_error']=angle(a.direction,vel[1:]/np.linalg.norm(vel[1:]))
            row['finite_1000M_vs_infinity_direction_error']=angle(a.direction,b.direction)
        single.append(row)
    result['single_mass_zero']=single
    result['passed']=(signature_failures==0 and max(inversion)<1e-12 and max(derivative)<1e-7
                      and all(x['binary_status']==x['schwarzschild_status'] and x['max_state_component_difference']<1e-6 for x in single)
                      and all(len(set(x['status'] for x in row['runs']))==1 and row['runs'][-1]['status']!='unresolved' for row in convergences)
                      and all(row['runs'][1].get('source_angle_from_tight',0)<1e-6 for row in convergences)
                      and all(v['successive_direction_changes_radians'][2]<v['successive_direction_changes_radians'][1]/2 for v in radius_rows))
    write(output/'validation.json',result)
    if not result['passed']:raise RuntimeError('binary validation failed; inspect validation.json')
    return result


def preview(metric,camera,settings,output,workers):
    output=Path(output)
    report=render(metric,camera,settings,output/'starfield',background=Starfield(),workers=workers)
    # Same saved trajectories: the grid does not perform another integration.
    grid=CelestialGrid()
    with np.load(output/'starfield/rays.npz') as z:
        image=shade(z['directions'],z['status'],grid,1)
    save_image(output/'grid.png',image)
    write(output/'grid_metadata.json',dict(source_ray_data='starfield/rays.npz',camera=report['camera'],
          spacetime=report['spacetime'],background=grid.description,additional_integrations=0,
          orientation='identical pixels and rays to starfield/shadow.png'))
    atlas=Image.new('RGB',(768,314),(246,247,249));draw=ImageDraw.Draw(atlas)
    for i,(file,label) in enumerate((('starfield/shadow.png','Astronomical starfield'),('grid.png','Celestial grid (same rays)'))):
        with Image.open(output/file) as im:atlas.paste(im.resize((384,288),Image.Resampling.NEAREST),(i*384,26))
        draw.text((i*384+8,7),label+' / experimental binary KS',fill=(15,20,30))
    atlas.save(output/'comparison.png')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path('examples/binary_ks_approx/equal_mass_12M.toml'))
    parser.add_argument('--output',type=Path,default=Path('output/binary-ks-experiment'))
    parser.add_argument('--render',action='store_true',help='96x72 default; traces once then repaints grid')
    parser.add_argument('--workers',type=int,default=1)
    args=parser.parse_args()
    m,c,s=load_configuration(args.config)
    audit(m,c,s,args.output)
    if args.render:preview(m,c,s,args.output,args.workers)


if __name__=='__main__':main()
