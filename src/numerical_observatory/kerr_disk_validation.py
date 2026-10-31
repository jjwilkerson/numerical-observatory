"""Reproduce validated small Kerr-disk previews and independent physical audits.

Use --render --workers 10 for six native 96x72 Kerr previews plus Schwarzschild
control. Default only audits selected rays/orbits. --figures-only never traces.
"""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import time
import numpy as np
from PIL import Image, ImageDraw
from .cli import load_configuration
from .disk import make_disk
from .kerr import Kerr
from .metric import Schwarzschild
from .camera import Camera
from .kerr_support import inclined_camera_options
from .emitter_motion import KerrCircular,kerr_isco
from .kerr_disk_reference import (independent_isco,circular_residual,circular_bl,
    circular_integrals,massive_radial,bl_frequency_shift,separated_hits)
from .kerr_reference import pixel_momentum
from .geodesic import Integration,trace
from .emission import frequency_shift
from .background import Starfield,CelestialGrid
from .render import render
from .surface_render import repaint_surface
from .disk_diagnostics import generate

CASES=('spin0_prograde_i85','spin05_prograde_i85','spin09_prograde_i85',
       'spin09_retrograde_i85','spin_minus09_prograde_i85','spin09_prograde_i60')
CONFIGS=Path('examples/kerr_disks/configs')
PROBES=[('crossing1',.9,'retrograde',(382.05182957538767,247.04245815155477)),
        ('crossing2',.9,'retrograde',(304.4455154940234,204.24704476156725)),
        ('crossing3',.9,'retrograde',(367.0613895063265,275.17830073795415)),
        ('crossing4',.9,'retrograde',(339.69077568795615,286.7805635605909)),
        ('approaching',.9,'prograde',(270.,247.5)),
        ('receding',.9,'prograde',(370.,247.5)),
        ('negative_spin',-.9,'prograde',(369.,247.5)),
        ('high_spin_ergosphere',.99,'prograde',(319.5,241.)),
        ('captured_without_disk',.9,'prograde',(327.5,247.5))]


def context(spin=.9,orbit='prograde',inclination=85):
    metric=Kerr(spin=spin)
    camera=Camera(**inclined_camera_options(metric,dict(radius=60,inclination_degrees=inclination,
                    width=640,height=480,fov_degrees=60)))
    settings=Integration(rtol=1e-11,atol=1e-13,escape_radius=110,max_affine=400,max_step=2)
    return metric,camera,settings,make_disk(metric,dict(enabled=True,orbit=orbit,display=dict(palette='blackbody',exposure=5)))


def physical_audit(output):
    emitters=[]
    for spin in (0.,.5,-.5,.9,-.9,.99,-.99):
        for orbit in ('prograde','retrograde'):
            metric=Kerr(spin=spin); motion=KerrCircular(metric,orbit)
            isco=kerr_isco(metric,orbit); independent=independent_isco(metric,orbit)
            errors=[]
            for r in (isco,isco*1.2,20.):
                for phi in (0.,.7,2.4):
                    rho=np.sqrt(r*r+metric.a**2); e=np.array([0.,rho*np.cos(phi),rho*np.sin(phi),0.])
                    u=motion.four_velocity(e); p=metric.covariant(e)@u
                    omega,ut,E,L=circular_bl(metric,r,orbit)
                    errors.append([abs(u@p+1),circular_residual(metric,motion,e),
                                   abs(-p[0]-E),abs(e[1]*p[2]-e[2]*p[1]-L),
                                   abs(motion.omega(r)-omega)])
            maximum=np.max(errors,axis=0)
            if max(maximum)>1e-11 or abs(isco-independent)>1e-10:
                raise RuntimeError('independent Kerr circular-orbit validation failed')
            emitters.append(dict(spin=spin,orbit=orbit,rotation=motion.rotation,isco_M=isco,
                independent_isco_M=independent,isco_difference_M=isco-independent,
                max_normalization_error=float(maximum[0]),max_geodesic_residual=float(maximum[1]),
                max_energy_difference=float(maximum[2]),max_Lz_difference=float(maximum[3]),
                max_omega_difference=float(maximum[4])))
    photons=[]
    for name,spin,orbit,xy in PROBES:
        metric,camera,settings,disk=context(spin,orbit)
        p=pixel_momentum(metric,camera,*xy)
        result=trace(metric,camera.event,p,settings,surface=disk.surface)
        if result.status!='emitted': raise RuntimeError(f'{name}: expected emitting test ray, got {result.status}')
        hit=result.hit
        independent=separated_hits(metric,camera,p,hit.affine+.05)
        first=next((h for h in independent if disk.surface.contains(h.event)),None)
        if first is None or first.crossing_index!=hit.crossing_index:
            raise RuntimeError(f'{name}: independent first-hit mismatch')
        g=frequency_shift(p,camera.tetrad(metric)[0],hit.momentum,disk.motion.four_velocity(hit.event))
        g_ref=bl_frequency_shift(metric,camera,p,metric.radius(hit.event),orbit)
        event_error=float(np.max(abs(hit.event-first.event))); p_error=float(np.max(abs(hit.momentum-first.momentum)))
        # Seventh-crossing shots are far more sensitive to initial/ODE error;
        # expose their tighter-tolerance comparison rather than hide that fact.
        tighter=trace(metric,camera.event,p,replace(settings,rtol=1e-12,atol=1e-14,max_step=1),surface=disk.surface)
        if tighter.status!='emitted' or tighter.hit.crossing_index!=hit.crossing_index:
            raise RuntimeError(f'{name}: nonconvergent emitting classification')
        convergence=float(np.max(abs(hit.event-tighter.hit.event)))
        if event_error>3e-4 or p_error>3e-5 or abs(g/g_ref-1)>1e-6:
            raise RuntimeError(f'{name}: independent photon-intersection disagreement')
        photons.append(dict(name=name,spin=spin,orbit=orbit,pixel=list(xy),crossing=hit.crossing_index,
            affine=hit.affine,event=hit.event.tolist(),radius_M=metric.radius(hit.event),
            azimuth_degrees=float(np.degrees(np.arctan2(hit.event[2],hit.event[1]))%360),
            preceding_crossing_radii_M=[metric.radius(h.event) for h in independent if h.crossing_index<hit.crossing_index],
            g=g,independent_BL_g=g_ref,relative_g_error=abs(g/g_ref-1),
            max_independent_event_difference_M=event_error,max_independent_covector_difference=p_error,
            max_tighter_event_difference_M=convergence,null_error=result.max_null_error,
            E_drift=result.energy_drift,Lz_drift=result.angular_momentum_drift,Q_drift=result.carter_drift))
        print(name,'crossing',hit.crossing_index,'r',metric.radius(hit.event),'g',g,'event error',event_error,flush=True)
    result=dict(emitter_validation=emitters,photon_validation=photons,
                geometry='equatorial z=0; oblate Kerr radial coordinate; first valid opaque hit',
                note='Independent BL radial equilibrium, marginal-stability radial potential, and exterior Carter-separated photon intersections; crossing ordinal is not winding number')
    (output/'physical_validation.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    rotated_zero_spin_audit(output)
    return result


def rotated_zero_spin_audit(output):
    """Explicitly rotate the legacy tilted-disk/+Z-camera Schwarzschild setup."""
    m,c,s,d=context(spin=0); sm=Schwarzschild(); i=np.deg2rad(85)
    C,S=np.cos(i),np.sin(i)
    R=np.array([[0.,-C,S],[1.,0.,0.],[0.,S,C]])
    sc=Camera(position=(0,0,60),width=640,height=480)
    sd=make_disk(sm,dict(enabled=True,inclination_degrees=85,display=dict(palette='blackbody',exposure=5)))
    rows=[]
    for x,y in [(420,232),(310,192),(351,276),(287,274),(319,200),(0,0)]:
        p=pixel_momentum(m,c,x,y); q=pixel_momentum(sm,sc,x,y)
        a=trace(m,c.event,p,s,surface=d.surface); b=trace(sm,sc.event,q,s,surface=sd.surface)
        if a.status!=b.status: raise RuntimeError('rotated zero-spin status mismatch')
        row=dict(pixel=[x,y],kerr_status=a.status,schwarzschild_status=b.status,
            initial_momentum_rotation_error=float(np.max(abs(p-np.r_[q[0],R@q[1:]]))))
        if a.hit:
            if a.hit.crossing_index!=b.hit.crossing_index: raise RuntimeError('rotated crossing mismatch')
            ua=d.motion.four_velocity(a.hit.event); ub=sd.motion.four_velocity(b.hit.event)
            ga=frequency_shift(p,c.tetrad(m)[0],a.hit.momentum,ua)
            gb=frequency_shift(q,sc.tetrad(sm)[0],b.hit.momentum,ub)
            Ia=d.emission.observed_bolometric(a.hit.event,ga); Ib=sd.emission.observed_bolometric(b.hit.event,gb)
            ca=d.display.linear_rgb(Ia,ga*d.emission.temperature(a.hit.event))
            cb=sd.display.linear_rgb(Ib,gb*sd.emission.temperature(b.hit.event))
            row.update(kerr_crossing=a.hit.crossing_index,schwarzschild_crossing=b.hit.crossing_index,
                max_position_rotation_error_M=float(np.max(abs(a.hit.event[1:]-R@b.hit.event[1:]))),
                max_velocity_rotation_error=float(np.max(abs(ua-np.r_[ub[0],R@ub[1:]]))),
                g_difference=abs(ga-gb),relative_bolometric_difference=abs(Ia/Ib-1),
                max_linear_RGB_difference=float(np.max(abs(ca-cb))))
            if row['max_position_rotation_error_M']>3e-6 or row['relative_bolometric_difference']>2e-6:
                raise RuntimeError('rotated zero-spin disk physics mismatch')
        elif a.direction is not None:
            row['source_direction_rotation_error']=float(np.linalg.norm(a.direction-R@b.direction))
        rows.append(row)
    record=dict(rotation_old_Schwarzschild_to_equatorial_Kerr=R.tolist(),inclination_degrees=85,
        note='Proper rotation maps the legacy tilted normal to +Z, camera and image axes together.',cases=rows)
    (output/'rotated_zero_spin_comparison.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
    return record


def nearcritical_reference_audit(output):
    """Optional precision investigation, not a certificate for a seventh image.

    Report failure against the SAME coordinate criterion. Never widen it to
    make an extreme reference comparison pass and never force a null rejection.
    """
    metric,camera,settings,disk=context(.9,'retrograde')
    xy=(314.5056064910384,280.9199781195458)
    p=pixel_momentum(metric,camera,*xy)
    production=[]
    for tol in (1e-11,1e-12,1e-13):
        result=trace(metric,camera.event,p,replace(settings,rtol=tol,atol=tol/100,max_step=1),surface=disk.surface)
        production.append(dict(rtol=tol,status=result.status,null_error=result.max_null_error,
            crossing=None if result.hit is None else result.hit.crossing_index,
            affine=None if result.hit is None else result.hit.affine,
            event=None if result.hit is None else result.hit.event.tolist()))
    references=[]
    if production[-1]['event'] is not None:
        duration=production[-1]['affine']+.1
        for tol in (1e-12,1e-13,2.5e-14):
            hits=separated_hits(metric,camera,p,duration,rtol=tol,atol=tol/10)
            first=next((h for h in hits if disk.surface.contains(h.event)),None)
            error=None if first is None else float(np.max(abs(np.asarray(production[-1]['event'])-first.event)))
            references.append(dict(rtol=tol,crossing=None if first is None else first.crossing_index,
                affine=None if first is None else first.affine,max_event_difference_M=error,
                coordinate_check_passed=bool(error is not None and error<=3e-4 and first.crossing_index==production[-1]['crossing'])))
    record=dict(pixel=list(xy),spin=.9,orbit='retrograde',production=production,separated_reference=references,
        coordinate_agreement_tolerance_M=3e-4,
        independent_coordinate_check_passed=bool(references and references[-1]['coordinate_check_passed']),
        note='Optional extreme near-critical probe. The existing double-precision second-order separated reference does not certify this event if its coordinate check fails. Production/null and reference convergence are reported separately. No criterion is relaxed; core validation covers first through fourth crossings.')
    (output/'nearcritical_reference_limit.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
    return record


def compare_zero_spin(output):
    with np.load(output/'schwarzschild_i85/disk_rays.npz') as a,np.load(output/'spin0_prograde_i85/disk_rays.npz') as b:
        emitted=(a['status']==3)&(b['status']==3)
        result=dict(status_differences=int(np.count_nonzero(a['status']!=b['status'])),
            crossing_differences=int(np.count_nonzero(a['crossing_index']!=b['crossing_index'])),
            max_position_difference_M=float(np.nanmax(abs(a['hit_position'][emitted]-b['hit_position'][emitted]))),
            max_radius_difference_M=float(np.nanmax(abs(a['hit_radius'][emitted]-b['hit_radius'][emitted]))),
            max_g_difference=float(np.nanmax(abs(a['g'][emitted]-b['g'][emitted]))),
            max_relative_bolometric_difference=float(np.nanmax(abs(a['bolometric_intensity'][emitted]/b['bolometric_intensity'][emitted]-1))))
        if result['status_differences'] or result['crossing_differences'] or result['max_position_difference_M']>1e-4 or result['max_g_difference']>2e-6:
            raise RuntimeError('zero-spin disk preview disagrees with Schwarzschild reference: '+str(result))
    with Image.open(output/'schwarzschild_i85/shadow.png') as a,Image.open(output/'spin0_prograde_i85/shadow.png') as b:
        difference=abs(np.asarray(a,dtype=int)-np.asarray(b,dtype=int))
        result.update(max_display_8bit_difference=int(difference.max()),different_display_pixels=int(np.count_nonzero(difference.any(axis=-1))))
    (output/'zero_spin_comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def compare_spin_reversal(output):
    with np.load(output/'spin09_prograde_i85/disk_rays.npz') as a,np.load(output/'spin_minus09_prograde_i85/disk_rays.npz') as b:
        mirrored=b['status'][:,::-1]
        emitted=(a['status']==3)&(mirrored==3)
        record=dict(status_differences=int(np.count_nonzero(a['status']!=mirrored)),
            crossing_differences=int(np.count_nonzero(a['crossing_index']!=b['crossing_index'][:,::-1])),
            max_g_difference=float(np.nanmax(abs(a['g'][emitted]-b['g'][:,::-1][emitted]))),
            max_position_reflection_difference_M=float(np.nanmax(abs(a['hit_position'][emitted]-b['hit_position'][:,::-1][emitted]*np.array([1,-1,1])))),
            max_relative_bolometric_difference=float(np.nanmax(abs(a['bolometric_intensity'][emitted]/b['bolometric_intensity'][:,::-1][emitted]-1))),
            note='Fixed +Z disk normal and identical camera; reflect image columns and global Y, reverse spin and prograde coordinate rotation. Photographic background need not be color-mirror symmetric.')
    (output/'spin_reversal_comparison.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
    return record


def cropped_comparison(output):
    """Same native crop in each required comparison panel; no new geodesics."""
    names=('schwarzschild_i85','spin05_prograde_i85','spin09_prograde_i85','spin09_retrograde_i85')
    labels=('Schwarzschild; inner edge 6.000M; i=85 deg',
            'Kerr a*=0.5 prograde; inner edge 4.233M',
            'Kerr a*=0.9 prograde; inner edge 2.321M',
            'Kerr a*=0.9 retrograde; inner edge 8.717M')
    bounds=(16,12,80,60)
    atlas=Image.new('RGB',(768,632),(246,247,249)); draw=ImageDraw.Draw(atlas)
    for i,name in enumerate(names):
        with Image.open(output/name/'shadow.png') as im:
            if im.size!=(96,72): raise ValueError('documented preview crop requires the matched 96x72 renders')
            image=im.crop(bounds).convert('RGB').resize((384,288),Image.Resampling.NEAREST)
        x=(i%2)*384; y=(i//2)*316
        atlas.paste(image,(x,y+28)); draw.text((x+6,y+6),labels[i],fill=(15,20,30))
    atlas.save(output/'central_comparison.png')
    (output/'comparison_crop.json').write_text(json.dumps(dict(source_bounds_xyxy_exclusive=list(bounds),
        original_size=[96,72],enlargement=6,interpolation='nearest neighbor; no extra angular sampling',cases=list(names)),indent=2)+'\n')


def figures(output):
    for name in ('schwarzschild_i85',)+CASES:
        repaint_surface(output/name,output/'grid'/name,background=CelestialGrid())
        generate(output/name,output/'diagnostics'/name)
    for filename,names,columns in [('comparison.png',('schwarzschild_i85','spin05_prograde_i85','spin09_prograde_i85','spin09_retrograde_i85'),2),
                                   ('all_kerr_previews.png',CASES,3)]:
        size=(384,288) if columns==2 else (288,216)
        atlas=Image.new('RGB',(columns*size[0],((len(names)+columns-1)//columns)*(size[1]+28)),(246,247,249))
        draw=ImageDraw.Draw(atlas)
        for i,name in enumerate(names):
            meta=json.loads((output/name/'diagnostics.json').read_text())
            with Image.open(output/name/'shadow.png') as image:
                image=image.convert('RGB').resize(size,Image.Resampling.NEAREST)
            x=(i%columns)*size[0]; y=(i//columns)*(size[1]+28)
            atlas.paste(image,(x,y+28))
            inner=meta['geometry']['inner_radius']/meta['mass']
            draw.text((x+6,y+6),f'{name}; r_in={inner:.3f}M',fill=(15,20,30))
        atlas.save(output/filename)
    compare_zero_spin(output)
    compare_spin_reversal(output)
    cropped_comparison(output)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('examples/kerr_disks'))
    parser.add_argument('--workers',type=int,default=1)
    parser.add_argument('--render',action='store_true')
    parser.add_argument('--figures-only',action='store_true')
    parser.add_argument('--nearcritical-audit',action='store_true',help='optional seventh-crossing reference precision experiment; reports failures explicitly')
    args=parser.parse_args(); args.output.mkdir(parents=True,exist_ok=True)
    if args.nearcritical_audit:
        print(json.dumps(nearcritical_reference_audit(args.output),indent=2),flush=True)
        if not args.render and not args.figures_only:
            return
    if not args.figures_only: physical_audit(args.output)
    if args.render:
        for name in ('schwarzschild_i85',)+CASES:
            config=CONFIGS/((CASES[0] if name=='schwarzschild_i85' else name)+'.toml')
            metric,camera,settings=load_configuration(config)
            options=__import__('tomllib').loads(config.read_text())['disk']
            if name=='schwarzschild_i85':
                metric=Schwarzschild(); options={k:v for k,v in options.items() if k!='orbit'}
                options.update(inner_radius=6.,inclination_degrees=0.)
            disk=make_disk(metric,options)
            start=time.perf_counter()
            report=render(metric,camera,settings,args.output/name,disk=disk,background=Starfield(),workers=args.workers)
            print(name,report['counts'],'elapsed',time.perf_counter()-start,flush=True)
    if args.render or args.figures_only: figures(args.output)


if __name__=='__main__': main()
