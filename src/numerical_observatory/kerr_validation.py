"""Reproduce small Kerr examples and quantitative validation (CPU only).

python -m numerical_observatory.kerr_validation --render --workers 10
Without --render only the numerical audits run; no existing source is overwritten.
"""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import time
import numpy as np
from PIL import Image,ImageDraw
from .cli import load_configuration
from .kerr import Kerr
from .camera import Camera
from .geodesic import trace,Integration
from .kerr_reference import (shadow_curve,pixel_momentum,separated_reference,
                             reference_radius,locate_boundary)
from .background import Starfield,CelestialGrid
from .render import render

CASES=('spin0_i90','spin05_i90','spin09_i90','spin09_i60','spin09_i20','spin_minus09_i90')


def configuration(name):
    return load_configuration(Path('examples/kerr')/(name+'.toml'))


def make_figures(output):
    atlas=Image.new('RGB',(3*256,2*216),(246,247,249)); draw=ImageDraw.Draw(atlas)
    gridatlas=atlas.copy(); gd=ImageDraw.Draw(gridatlas)
    curves={}
    for i,name in enumerate(CASES):
        m,c,_=configuration(name)
        curve=shadow_curve(m,c,20001); curves[name]=curve
        with Image.open(output/name/'shadow.png') as im:
            im=im.convert('RGB').resize((256,192),Image.Resampling.NEAREST)
        x=(i%3)*256; y=(i//3)*216
        atlas.paste(im,(x,y+24)); draw.text((x+5,y+5),f'a*={m.spin:g}, inclination={np.rad2deg(np.arccos(c.position[2]/m.radius(c.event))):g} deg',fill=(15,20,30))
        overlay=im.copy(); od=ImageDraw.Draw(overlay)
        points=[((p[0]+.5)*4-.5,(p[1]+.5)*4-.5) for p in curve]
        od.line(points+[points[0]],fill=(60,255,130),width=1)
        overlay.save(output/name/'shadow_theory_overlay.png')
        with np.load(output/name/'rays.npz') as data:
            status=data['status']; directions=data['directions']
            rgb=np.zeros((*status.shape,3))
            rgb[status==1]=CelestialGrid().color(directions[status==1])
            rgb[status==2]=(1.,0.,.7)
        grid=Image.fromarray(np.uint8(np.rint(np.clip(rgb,0,1)*255)))
        grid.save(output/name/'grid.png')
        gridatlas.paste(grid.resize((256,192),Image.Resampling.NEAREST),(x,y+24))
        gd.text((x+5,y+5),f'a*={m.spin:g}, i={np.rad2deg(np.arccos(c.position[2]/m.radius(c.event))):g} deg',fill=(15,20,30))
    atlas.save(output/'comparison.png'); gridatlas.save(output/'grid_comparison.png')
    np.savez_compressed(output/'analytical_curves.npz',**curves)


def audit_boundaries(output):
    all_cases={}
    for name in CASES:
        m,c,s=configuration(name); curve=shadow_curve(m,c,20001)
        angles=np.linspace(0,2*np.pi,12,endpoint=False)+.071
        rows=[]
        for angle in angles:
            radius=reference_radius(curve,c,angle)
            rows.append(dict(angle_radians=float(angle),reference_radius_pixels=radius,
                             **locate_boundary(m,c,s,angle,radius,precision=1e-4)))
        error=np.array([r['error_pixels'] for r in rows])
        if np.max(np.abs(error)) > 1e-3:
            raise RuntimeError(f'{name}: analytical shadow disagreement exceeds 0.001 pixel')
        extrema=dict(x_min=float(curve[:,0].min()),x_max=float(curve[:,0].max()),
                     y_min=float(curve[:,1].min()),y_max=float(curve[:,1].max()))
        all_cases[name]=dict(samples=rows,max_abs_error_pixels=float(max(abs(error))),rms_error_pixels=float(np.sqrt(np.mean(error**2))),
                             analytical_extrema_pixels=extrema,
                             center_displacement_pixels=(extrema['x_min']+extrema['x_max'])/2-(c.width-1)/2,
                             half_width_pixels=(extrema['x_max']-extrema['x_min'])/2,
                             half_height_pixels=(extrema['y_max']-extrema['y_min'])/2)
        print(name,'boundary max/rms',all_cases[name]['max_abs_error_pixels'],all_cases[name]['rms_error_pixels'],flush=True)
    (output/'shadow_validation.json').write_text(json.dumps(all_cases,indent=2,allow_nan=False)+'\n')
    return all_cases


def make_boundary_figure(output):
    """Repaint theory and measured subpixel boundaries; never retraces rays."""
    path=output/'shadow_validation.json'
    if not path.exists() or not all((output/name/'shadow.png').exists() for name in CASES):
        return
    data=json.loads(path.read_text())
    atlas=Image.new('RGB',(768,458),(246,247,249)); draw=ImageDraw.Draw(atlas)
    for i,name in enumerate(CASES):
        m,c,_=configuration(name)
        with Image.open(output/name/'shadow.png') as im:
            im=im.convert('RGB').resize((256,192),Image.Resampling.NEAREST)
        painter=ImageDraw.Draw(im)
        curve=shadow_curve(m,c,20001)
        points=[((p[0]+.5)*4-.5,(p[1]+.5)*4-.5) for p in curve]
        painter.line(points+[points[0]],fill=(60,255,130),width=1)
        center=np.array([(c.width-1)/2,(c.height-1)/2])
        for row in data[name]['samples']:
            angle=row['angle_radians']; radius=row['radius_pixels']
            p=center+radius*np.array([np.cos(angle),np.sin(angle)])
            x,y=(p+.5)*4-.5
            painter.ellipse((x-1.5,y-1.5,x+1.5,y+1.5),fill=(255,85,110))
        im.save(output/name/'shadow_theory_overlay.png')
        x=(i%3)*256; y=(i//3)*216
        atlas.paste(im,(x,y+24))
        draw.text((x+5,y+5),name,fill=(15,20,30))
    draw.text((8,438),'Green: finite-camera theory. Red: Hamiltonian capture/escape bisections (12 angles).',fill=(15,20,30))
    atlas.save(output/'shadow_validation_comparison.png')


def audit_convergence(output):
    m,c,s=configuration('spin09_i60'); curve=shadow_curve(m,c,40001)
    records=[]
    for tol in (1e-7,1e-9,1e-11):
        current=replace(s,rtol=tol,atol=tol/100)
        boundaries=[]
        for angle in (.3,1.2,2.3,3.8):
            try:
                boundaries.append(dict(angle_radians=angle,status="resolved",
                    **locate_boundary(m,c,current,angle,reference_radius(curve,c,angle),precision=1e-5)))
            except RuntimeError as exc:
                # A coarse near-critical shot may correctly fail the production
                # null guard. Retain this failure, rather than weakening the
                # guard or substituting an analytical classification.
                if 'unresolved boundary ray' not in str(exc):
                    raise
                boundaries.append(dict(angle_radians=angle,status="unresolved",reason=str(exc)))
        rows=[]
        for x,y in ((56,30),(48,6),(12,24)):
            p=pixel_momentum(m,c,x,y)
            result,sol=trace(m,c.event,p,current,return_solution=True)
            if result.status!='escaped': raise RuntimeError('audit expects escaping ray')
            ref=separated_reference(m,np.r_[c.event,p],sol.t[-1])
            if not ref.success: raise RuntimeError('independent separated reference failed')
            times=np.linspace(0,sol.t[-1],501); a=sol.sol(times).T; b=ref.sol(times).T
            radii=np.array([m.radius(z[:4]) for z in a])
            rows.append(dict(pixel=[x,y],status=result.status,null_error=result.max_null_error,
                             energy_drift=result.energy_drift,axial_momentum_drift=result.angular_momentum_drift,
                             carter_drift=result.carter_drift,
                             radial_turning=bool(b[:,1].min()<0<b[:,1].max()),
                             polar_turning=bool(b[:,3].min()<0<b[:,3].max()),
                             max_radius_difference_M=float(max(abs(radii-b[:,0]))),
                             max_theta_difference_radians=float(max(abs(np.arccos(a[:,3]/radii)-b[:,2]))),
                             max_phi_difference_radians=float(max(abs(np.angle(np.exp(1j*(np.arctan2(a[:,2],a[:,1])-b[:,4])))))),
                             max_time_difference_M=float(max(abs(a[:,0]-b[:,5])))))
        records.append(dict(rtol=tol,atol=tol/100,boundaries=boundaries,trajectories=rows))
        print('convergence',tol,flush=True)
    # Separate far-field truncation and capture-event policy from ODE error.
    p=pixel_momentum(m,c,56,30)
    source=[trace(m,c.event,p,replace(s,rtol=1e-11,atol=1e-13,kerr_source_radius=R)).direction for R in (1e4,1e5,1e6)]
    capture=[]
    for fraction in (.98,.995,.999):
        current=replace(s,kerr_capture_fraction=fraction)
        a=.7
        capture.append(dict(fraction=fraction,**locate_boundary(m,c,current,a,reference_radius(curve,c,a),precision=1e-5)))
    # a=0 compared with the independent pre-existing Schwarzschild ODE/tail.
    from .metric import Schwarzschild
    zero=Kerr(); _,c0,s0=configuration('spin0_i90'); limit=[]
    for x,y in ((31.5,23.5),(56,30),(49,10)):
        p=pixel_momentum(zero,c0,x,y)
        a,sa=trace(zero,c0.event,p,s0,return_solution=True)
        b,sb=trace(Schwarzschild(),c0.event,p,s0,return_solution=True)
        limit.append(dict(pixel=[x,y],kerr_status=a.status,schwarzschild_status=b.status,
                          max_state_difference=float(np.max(abs(sa.y-sb.y))),
                          source_direction_difference=None if a.direction is None else float(np.linalg.norm(a.direction-b.direction))))
    result=dict(convergence=records,
                far_field_direction_differences=dict(R1e4_vs_1e5=float(np.linalg.norm(source[0]-source[1])),R1e5_vs_1e6=float(np.linalg.norm(source[1]-source[2]))),
                capture_fraction_comparison=capture,schwarzschild_limit=limit,
                notes='Boundary bracket widths, curve interpolation, raster sampling and asymptotic-tail truncation are separate error sources. Carter-reference uses separated second-order Mino velocities in affine time, not production metric derivatives.')
    (output/'convergence.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('examples/kerr_validation'))
    parser.add_argument('--render',action='store_true')
    parser.add_argument('--workers',type=int,default=1)
    parser.add_argument('--figures-only',action='store_true',help='reuse already rendered ray files')
    args=parser.parse_args(); args.output.mkdir(parents=True,exist_ok=True)
    if args.render:
        for name in CASES:
            m,c,s=configuration(name)
            start=time.perf_counter()
            report=render(m,c,s,args.output/name,background=Starfield(),workers=args.workers)
            print(name,report['counts'],'wall',time.perf_counter()-start,flush=True)
    if args.render or args.figures_only: make_figures(args.output)
    if not args.figures_only:
        audit_boundaries(args.output)
        audit_convergence(args.output)
    make_boundary_figure(args.output)


if __name__=='__main__':
    main()
