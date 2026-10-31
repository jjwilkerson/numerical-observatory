"""Carter/Hamiltonian validation, shadow shooting, images and CPU benchmarks.

All products use separate directories. No production physics is redefined here.
"""
import argparse
from dataclasses import replace,asdict
from decimal import Decimal,localcontext
import json
from pathlib import Path
import time
import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from PIL import Image,ImageDraw
from .kerr import Kerr
from .camera import Camera
from .geodesic import Integration,trace,hamilton_rhs
from .kerr_support import inclined_camera_options,default_domain
from .kerr_reference import pixel_momentum,shadow_curve,reference_radius,separated_reference
from .kerr_carter_quadrature import escaping_crossing
from .kerr_carter import CarterOrbit
from .disk import make_disk
from .emission import frequency_shift
from .background import Starfield,CelestialGrid
from .render import render
from .diagnostic_render import shade,save_image

DEFAULT=Path('examples/kerr_carter')


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def scene(name,width=96,height=72):
    spin=.99 if name=='high_spin' else .9
    inc=60 if name in ('shadow','moderate') else 85
    metric=Kerr(spin=spin)
    camera=Camera(**inclined_camera_options(metric,dict(radius=60,inclination_degrees=inc,
                    width=width,height=height,fov_degrees=60)))
    settings=Integration(rtol=1e-9,atol=1e-11,escape_radius=110,max_affine=400,max_step=4)
    options=dict(enabled=True,orbit='retrograde' if name=='retrograde' else 'prograde',
                 display=dict(palette='blackbody',exposure=5))
    disk=None if name=='shadow' else make_disk(metric,options)
    return metric,camera,settings,disk


def boundary_sample(metric,camera,settings,angle,radius,precision=1e-4):
    """Capture/escape interval, allowing explicitly unresolved middle shots.

    A truly critical orbit has no finite capture/escape outcome. If a middle
    shot is unresolved, certify the requested interval ONLY from independently
    captured/escaped shots on either side; otherwise report an unresolved bound.
    """
    center=np.array([(camera.width-1)/2,(camera.height-1)/2]);unit=np.array([np.cos(angle),np.sin(angle)])
    domain=default_domain(metric,camera.event,settings);shots=[]
    def classify(r):
        result=trace(metric,camera.event,pixel_momentum(metric,camera,*(center+r*unit)),settings,domain=domain)
        shots.append(dict(radius_pixels=r,status=result.status,reason=result.reason,null=result.max_null_error,
                          final_backend=result.backend,fallback=result.fallback_reason))
        return result.status
    delta=max(.2,radius*.04)
    for _ in range(12):
        lo=max(.01,radius-delta);hi=radius+delta
        if classify(lo)=='captured' and classify(hi)=='escaped':break
        delta*=2
    else:raise RuntimeError('no capture/escape bracket')
    while hi-lo>precision:
        mid=(hi+lo)/2;status=classify(mid)
        if status=='captured':lo=mid
        elif status=='escaped':hi=mid
        else:
            left=mid-precision*.49;right=mid+precision*.49
            if classify(left)=='captured' and classify(right)=='escaped':
                lo,hi=left,right;break
            return dict(resolved=False,bracket_pixels=[lo,hi],shots=shots,
                        reason='uncertainty not bounded at requested precision')
    return dict(resolved=True,radius_pixels=(lo+hi)/2,reference_radius_pixels=radius,
                error_pixels=(lo+hi)/2-radius,bracket_pixels=[lo,hi],
                bracket_width_pixels=hi-lo,shots=shots,
                unresolved_shots=sum(x['status']=='unresolved' for x in shots))


def shadow_audit(output):
    records=[];curves={}
    for spin,inc in [(0,90),(.5,90),(.9,20),(.9,60),(-.9,85),(.99,85)]:
        m=Kerr(spin=spin)
        c=Camera(**inclined_camera_options(m,dict(radius=30,inclination_degrees=inc,width=96,height=72,fov_degrees=35)))
        s=Integration(rtol=1e-11,atol=1e-13,escape_radius=80,max_affine=350,max_step=4)
        curve=shadow_curve(m,c,20001);curves[f'spin{spin}_inc{inc}']=curve
        probes=[]
        for angle in np.linspace(0,2*np.pi,8,endpoint=False)+.071:
            radius=reference_radius(curve,c,angle);pair={}
            for backend in ('hamiltonian','carter'):
                pair[backend]=boundary_sample(m,c,replace(s,integrator=backend),float(angle),radius)
                if not pair[backend]['resolved']:raise RuntimeError('unresolved shadow interval: '+str((spin,inc,backend)))
                if abs(pair[backend]['error_pixels'])>1e-3:raise RuntimeError('analytical shadow mismatch')
            probes.append(dict(angle_radians=float(angle),**pair))
        item=dict(spin=spin,inclination_degrees=inc,camera=asdict(c),integration=asdict(s),samples=probes)
        for backend in ('hamiltonian','carter'):
            errors=np.array([x[backend]['error_pixels'] for x in probes])
            item[backend]=dict(max_abs_error_pixels=float(max(abs(errors))),rms_error_pixels=float(np.sqrt(np.mean(errors**2))),
                unresolved_shots=sum(x[backend]['unresolved_shots'] for x in probes))
        item['max_backend_disagreement_pixels']=max(abs(x['carter']['radius_pixels']-x['hamiltonian']['radius_pixels']) for x in probes)
        records.append(item);save(output/'shadow_validation.json',records)
        print('shadow',spin,inc,item['carter'],flush=True)
    np.savez_compressed(output/'shadow_curves.npz',**curves)
    return records


def ray_audit(output):
    records=[]
    for spin,inc in [(0,90),(.5,20),(.9,60),(-.9,85),(.99,85),(.9,90)]:
        m=Kerr(spin=spin)
        c=Camera(**inclined_camera_options(m,dict(radius=60,inclination_degrees=inc,width=640,height=480,fov_degrees=60)))
        s=Integration(rtol=1e-11,atol=1e-13,escape_radius=110,max_affine=400,max_step=4)
        disk=make_disk(m,dict(enabled=True))
        for xy in [(0,0),(270,247.5),(319,200),(370,247.5)]:
            p=pixel_momentum(m,c,*xy)
            h,hs=trace(m,c.event,p,s,surface=disk.surface,return_solution=True)
            a,cs=trace(m,c.event,p,replace(s,integrator='carter'),surface=disk.surface,return_solution=True)
            if h.status!=a.status or a.status=='unresolved':raise RuntimeError('ordinary ray status mismatch')
            times=np.linspace(0,min(hs.t[-1],cs.t[-1])*.999,41)
            states=cs.sol(times).T;reference=hs.sol(times).T
            position=float(np.max(abs(states[:,:4]-reference[:,:4])))
            momentum=float(np.max(abs(states[:,4:]-reference[:,4:])))
            velocity_residual=0.;momentum_residual=0.
            for t in times[::4]:
                y=cs.separated_solution.sol(t/m.mass);o=cs.carter_orbit
                state=o.cartesian(y);equation=hamilton_rhs(m,state)
                velocity_residual=max(velocity_residual,float(np.max(abs(o.tangent(y)-equation[:4]))))
                if 1-y[2]**2>.01:
                    rate=o.rhs(0,y);step=1e-4
                    derivative=(-o.cartesian(y+2*step*rate)+8*o.cartesian(y+step*rate)-8*o.cartesian(y-step*rate)+o.cartesian(y-2*step*rate))/(12*step*m.mass)
                    momentum_residual=max(momentum_residual,float(np.max(abs(derivative[4:]-equation[4:]))))
            row=dict(spin=spin,inclination=inc,pixel=list(xy),status=a.status,max_event_component_difference_M=position,
                max_covector_difference=momentum,max_velocity_equation_residual=velocity_residual,
                max_momentum_equation_residual=momentum_residual,carter_null=a.max_null_error,
                carter_E_error=a.energy_drift,carter_Lz_error=a.angular_momentum_drift,carter_Q_error=a.carter_drift,
                separated_constraints=a.carter_diagnostics)
            if a.hit:
                ga=frequency_shift(p,c.tetrad(m)[0],a.hit.momentum,disk.motion.four_velocity(a.hit.event))
                gh=frequency_shift(p,c.tetrad(m)[0],h.hit.momentum,disk.motion.four_velocity(h.hit.event))
                row.update(carter_crossing=a.hit.crossing_index,hamiltonian_crossing=h.hit.crossing_index,g_difference=abs(ga-gh),
                    relative_intensity_difference=abs(disk.emission.observed_bolometric(a.hit.event,ga)/disk.emission.observed_bolometric(h.hit.event,gh)-1))
            elif a.direction is not None:row['source_direction_difference']=float(np.linalg.norm(a.direction-h.direction))
            if position>3e-6 or momentum>5e-7 or velocity_residual>2e-7 or momentum_residual>2e-6:
                raise RuntimeError('ordinary trajectory equation mismatch: '+str(row))
            records.append(row)
    save(output/'ray_validation.json',records);return records


def seventh_audit(output):
    m=Kerr(spin=.9)
    c=Camera(**inclined_camera_options(m,dict(radius=60,inclination_degrees=85,width=640,height=480,fov_degrees=60)))
    p=pixel_momentum(m,c,314.5056064910384,280.9199781195458)
    d=make_disk(m,dict(enabled=True,orbit='retrograde'))
    s=Integration(escape_radius=110,max_affine=400,max_step=1)
    quadratures=[]
    for precision,eps in [(30,1e-10),(50,2e-12),(70,5e-13)]:
        hit,info=escaping_crossing(m,c.event,p,7,precision=precision,eps=eps)
        quadratures.append(dict(event=hit.event.tolist(),momentum=hit.momentum.tolist(),affine=hit.affine,**info))
    oracle=np.array(quadratures[-1]['event']);runs=[]
    for tol in (1e-9,1e-11,1e-12,1e-13):
        for backend in ('hamiltonian','carter'):
            result,sol=trace(m,c.event,p,replace(s,integrator=backend,rtol=tol,atol=tol/100),surface=d.surface,return_solution=True)
            if result.status!='emitted' or result.hit.crossing_index!=7:raise RuntimeError('seventh classification did not converge')
            runs.append(dict(backend=backend,rtol=tol,event=result.hit.event.tolist(),affine=result.hit.affine,
                max_event_difference_from_quadrature=float(np.max(abs(result.hit.event-oracle))),null_error=result.max_null_error,
                Q_error=result.carter_drift,separated_diagnostics=result.carter_diagnostics))
    E,L,Q=m.invariants(np.r_[c.event,p]);old=[]
    for tol in (1e-12,1e-13,2.5e-14):
        ref=separated_reference(m,np.r_[c.event,p],100.,rtol=tol,atol=tol/10)
        if not ref.success:raise RuntimeError('old separated audit failed')
        ts=np.sort(np.r_[ref.t,(ref.t[:-1]+ref.t[1:])/2]);levels=np.cos(ref.sol(ts)[2]);roots=[]
        for lo,hi,fl,fh in zip(ts[:-1],ts[1:],levels[:-1],levels[1:]):
            if fl*fh<0:roots.append(brentq(lambda t:np.cos(ref.sol(t)[2]),lo,hi,xtol=1e-11))
        t=roots[6];r,v,theta,w,phi,T=ref.sol(t)
        event=np.array([T,np.sqrt(r*r+m.a*m.a)*np.sin(theta)*np.cos(phi),np.sqrt(r*r+m.a*m.a)*np.sin(theta)*np.sin(phi),r*np.cos(theta)])
        points=np.unique(np.r_[np.linspace(0,t,1001),ref.t[ref.t<t],t]);ys=ref.sol(points)
        radial=[];polar=[]
        with localcontext() as ctx:
            ctx.prec=50
            ed,ld,qd,ad=[Decimal.from_float(float(x)) for x in (E,L,Q,m.a)]
            for rr,vv,tt,ww in zip(ys[0],ys[1],ys[2],ys[3]):
                rd=Decimal.from_float(float(rr));vd=Decimal.from_float(float(vv))
                R=(ed*(rd*rd+ad*ad)-ad*ld)**2-(rd*rd-2*rd+ad*ad)*(qd+(ld-ad*ed)**2)
                radial.append(float(vd*vd-R)/(E*E))
                Theta=Q+m.a*m.a*E*E*np.cos(tt)**2-L*L/np.tan(tt)**2
                polar.append((ww*ww-Theta)/(E*E))
        tau=quad(lambda x:abs(E)/(ref.sol(x)[0]**2+m.a*m.a*np.cos(ref.sol(x)[2])**2),0,t,
                 points=ref.t[(ref.t>0)&(ref.t<t)],epsabs=1e-11,epsrel=1e-11,limit=max(1000,len(ref.t)+20))[0]
        closest=int(np.argmin(ys[0]));old.append(dict(rtol=tol,event=event.tolist(),affine=t,
            max_event_difference_from_quadrature=float(np.max(abs(event-oracle))),
            radial_constraint_initial=radial[0],radial_constraint_near_turn=radial[closest],
            max_radial_constraint=float(max(abs(np.array(radial)))),max_polar_constraint=float(max(abs(np.array(polar)))),
            mino_phase_to_crossing=tau,phase_difference_from_quadrature=tau-quadratures[-1]['mino_phase']))
    accepted=[x for x in runs if x['rtol']<=1e-11]
    if any(x['max_event_difference_from_quadrature']>3e-4 for x in accepted):
        raise RuntimeError('unchanged seventh-crossing acceptance criterion failed')
    save(output/'seventh_crossing.json',dict(original_criterion_M=3e-4,quadrature=quadratures,
        production_comparisons=runs,old_reference=old,
        camera=asdict(c),initial_momentum=p.tolist(),integration=asdict(s),
        original_record='examples/kerr_disks/nearcritical_reference_limit.json (preserved)',
        verdict='Both production backends pass the unchanged criterion at rtol<=1e-11 against independent regularized separated quadrature. The original r-variable reference retains its documented failure.'))
    return runs


def compare_outputs(ham,carter):
    ham,carter=Path(ham),Path(carter)
    filename='disk_rays.npz' if (ham/'disk_rays.npz').exists() else 'rays.npz'
    result={}
    with np.load(ham/filename) as h,np.load(carter/filename) as c:
        result['status_differences']=int(np.count_nonzero(h['status']!=c['status']))
        result['initial_constant_differences']=int(np.count_nonzero(h['kerr_constants']!=c['kerr_constants']))
        result['carter_unresolved']=int(np.count_nonzero(c['status']==2))
        escaped=(h['status']==1)&(c['status']==1)
        if escaped.any():
            a=h['directions'][escaped];b=c['directions'][escaped]
            angles=np.arctan2(np.linalg.norm(np.cross(a,b),axis=-1),np.sum(a*b,axis=-1))
            result['max_source_angle_radians']=float(angles.max())
        if filename=='disk_rays.npz':
            result['crossing_differences']=int(np.count_nonzero(h['crossing_index']!=c['crossing_index']))
            emitted=(h['status']==3)&(c['status']==3)
            result['max_hit_position_difference_M']=float(np.max(abs(h['hit_position'][emitted]-c['hit_position'][emitted])))
            result['max_hit_event_component_difference_M']=float(np.max(abs(h['hit_event'][emitted]-c['hit_event'][emitted])))
            result['max_emission_radius_difference_M']=float(np.max(abs(h['hit_radius'][emitted]-c['hit_radius'][emitted])))
            azimuth=np.deg2rad(h['emission_azimuth_degrees'][emitted]-c['emission_azimuth_degrees'][emitted])
            result['max_emission_azimuth_difference_degrees']=float(np.rad2deg(np.max(abs(np.angle(np.exp(1j*azimuth))))))
            result['max_g_difference']=float(np.max(abs(h['g'][emitted]-c['g'][emitted])))
            result['max_relative_bolometric_difference']=float(np.max(abs(h['bolometric_intensity'][emitted]/c['bolometric_intensity'][emitted]-1)))
    with Image.open(ham/'shadow.png') as a,Image.open(carter/'shadow.png') as b:
        difference=abs(np.asarray(a,dtype=int)-np.asarray(b,dtype=int))
        result.update(max_png_difference=int(difference.max()),different_png_pixels=int(np.count_nonzero(difference.any(axis=-1))))
    result['passed']=(result['status_differences']==0 and result['initial_constant_differences']==0 and result.get('crossing_differences',0)==0
        and result['carter_unresolved']==0 and result.get('max_hit_position_difference_M',0)<1e-4
        and result.get('max_g_difference',0)<2e-6 and result.get('max_relative_bolometric_difference',0)<2e-5
        and result.get('max_source_angle_radians',0)<2e-6)
    return result


def run_pair(root,name,width,height,workers,repeat):
    metric,camera,settings,disk=scene(name,width,height);rows=[]
    for backend in ('hamiltonian','carter'):
        output=root/f'{name}_{width}x{height}_{repeat}'/backend
        start=time.perf_counter();cpu=time.process_time()
        report=render(metric,camera,replace(settings,integrator=backend),output,disk=disk,background=Starfield(),workers=workers)
        wall=time.perf_counter()-start;parent_cpu=time.process_time()-cpu
        execution=report.get('execution',{})
        rows.append(dict(backend=backend,spin=metric.spin,inclination_degrees=60 if name in ('shadow','moderate') else 85,
            resolution=[width,height],rays=width*height,workers=workers,wall_seconds=wall,
            worker_cpu_seconds=execution.get('worker_cpu_seconds'),parent_cpu_seconds=parent_cpu,
            startup_span_seconds=execution.get('startup_span_seconds',0),rhs_evaluations=report['rhs_evaluations'],
            steps=report['total_steps'],unresolved=report['counts']['unresolved'],fallbacks=report.get('fallback_count',0),
            peak_worker_memory_bytes=execution.get('worker_peak_memory_bytes'),output=str(output),
            conservation=report.get('conservation_maxima',report.get('maxima'))))
        print('render',name,width,height,backend,'wall',wall,'unresolved',report['counts']['unresolved'],flush=True)
    comparison=compare_outputs(rows[0]['output'],rows[1]['output'])
    item=dict(scene=name,repeat=repeat,runs=rows,numerical_comparison=comparison,
              wall_speedup=rows[0]['wall_seconds']/rows[1]['wall_seconds'])
    if rows[0]['worker_cpu_seconds'] and rows[1]['worker_cpu_seconds']:
        item['worker_cpu_speedup']=rows[0]['worker_cpu_seconds']/rows[1]['worker_cpu_seconds']
        item['startup_adjusted_wall_speedup']=(rows[0]['wall_seconds']-rows[0]['startup_span_seconds'])/(rows[1]['wall_seconds']-rows[1]['startup_span_seconds'])
    save(root/f'{name}_{width}x{height}_{repeat}'/'comparison.json',item)
    if not comparison['passed']:raise RuntimeError('image/ray equivalence failed: '+str(item))
    return item


def benchmarks(output,workers=10):
    results=[]
    for name in ('shadow','edge','moderate'):
        results.append(run_pair(output/'benchmark',name,96,72,workers,0))
        save(output/'benchmarks.json',results)
    # Repeated small fixed-quality runs quantify variability separately. Worker
    # CPU ratios exclude spawn startup; do not call their raw wall ratio a pure
    # solver improvement when startup is a large fraction of a small render.
    for repeat in (1,2):
        for name in ('shadow','edge','moderate'):
            results.append(run_pair(output/'benchmark',name,32,24,workers,repeat))
            save(output/'benchmarks.json',results)
    return results


def small_images(output,workers=10):
    rows=[]
    for name in ('shadow','edge','retrograde','high_spin'):
        pair=run_pair(output/'images',name,32,24,workers,0);rows.append(pair)
        if name=='shadow':
            for backend in ('hamiltonian','carter'):
                path=Path(pair['runs'][0 if backend=='hamiltonian' else 1]['output'])
                with np.load(path/'rays.npz') as z:
                    save_image(path/'grid.png',shade(z['directions'],z['status'],CelestialGrid(),1))
    save(output/'image_comparisons.json',rows)
    panels=[]
    for pair in rows:
        for row in pair['runs']:
            with Image.open(Path(row['output'])/'shadow.png') as im:
                panels.append((im.convert('RGB').resize((320,240),Image.Resampling.NEAREST),pair['scene']+' / '+row['backend']))
    atlas=Image.new('RGB',(640,4*266),(246,247,249));draw=ImageDraw.Draw(atlas)
    for i,(im,title) in enumerate(panels):
        x=(i%2)*320;y=(i//2)*266
        atlas.paste(im,(x,y+26));draw.text((x+6,y+6),title,fill=(15,20,30))
    atlas.save(output/'comparison.png')
    figures_from_saved(output)
    return rows


def figures_from_saved(output):
    """Figures from already computed numerical data, no trajectory integration."""
    if (output/'image_comparisons.json').exists():
        records=json.loads((output/'image_comparisons.json').read_text())
        shadow=next(row for row in records if row['scene']=='shadow')
        atlas=Image.new('RGB',(640,266),(246,247,249));draw=ImageDraw.Draw(atlas)
        values=[]
        for i,row in enumerate(shadow['runs']):
            with Image.open(Path(row['output'])/'grid.png') as im:
                values.append(np.asarray(im).copy())
                atlas.paste(im.convert('RGB').resize((320,240),Image.Resampling.NEAREST),(320*i,26))
            draw.text((320*i+6,6),'Celestial grid / '+row['backend'],fill=(15,20,30))
        difference=abs(values[0].astype(int)-values[1].astype(int))
        shadow['grid_comparison']=dict(max_png_difference=int(difference.max()),different_pixels=int(np.count_nonzero(difference.any(axis=-1))))
        save(output/'image_comparisons.json',records)
        atlas.save(output/'grid_comparison.png')
    if (output/'shadow_validation.json').exists():
        records=json.loads((output/'shadow_validation.json').read_text())
        atlas=Image.new('RGB',(1152,652),(246,247,249));draw=ImageDraw.Draw(atlas)
        with np.load(output/'shadow_curves.npz') as curves:
            for i,row in enumerate(records):
                panel=Image.new('RGB',(384,288),(24,29,37));pd=ImageDraw.Draw(panel)
                curve=curves[f"spin{row['spin']}_inc{row['inclination_degrees']}"]
                points=[tuple((p+.5)*4-.5) for p in curve]
                pd.line(points+[points[0]],fill=(70,230,120),width=1)
                for sample in row['samples']:
                    angle=sample['angle_radians'];unit=np.array([np.cos(angle),np.sin(angle)])
                    for backend,color,radius in [('hamiltonian',(255,90,90),3),('carter',(70,160,255),1)]:
                        p=(np.array([47.5,35.5])+sample[backend]['radius_pixels']*unit+.5)*4-.5
                        pd.ellipse((p[0]-radius,p[1]-radius,p[0]+radius,p[1]+radius),outline=color,width=1)
                x=(i%3)*384;y=(i//3)*314
                atlas.paste(panel,(x,y+26));draw.text((x+6,y+6),f"a*={row['spin']}, i={row['inclination_degrees']} deg",fill=(15,20,30))
        draw.text((8,632),'Green: finite-camera theory. Red circles: Hamiltonian. Blue dots: Carter. No positional offsets.',fill=(15,20,30))
        atlas.save(output/'shadow_comparison.png')
    if (output/'seventh_crossing.json').exists():
        record=json.loads((output/'seventh_crossing.json').read_text())
        plot=Image.new('RGB',(800,530),(250,250,252));pd=ImageDraw.Draw(plot)
        def project(tol,error):return (80+(-np.log10(tol)-9)/5*650,450-(np.log10(error)+9)/8*370)
        for exponent in range(9,15):
            x,_=project(10.**-exponent,1e-9);pd.line((x,80,x,450),fill=(215,220,225));pd.text((x-16,462),f'1e-{exponent}',fill=(20,20,25))
        for exponent in range(1,10):
            _,y=project(1e-9,10.**-exponent);pd.line((80,y,730,y),fill=(215,220,225));pd.text((25,y-5),f'1e-{exponent}',fill=(20,20,25))
        _,limit=project(1e-9,record['original_criterion_M'])
        pd.line((80,limit,730,limit),fill=(190,70,70),width=2)
        pd.text((85,limit-16),'Unchanged 0.0003M criterion',fill=(150,40,40))
        colors={'hamiltonian':(200,70,50),'carter':(35,100,205),'old reference':(130,90,160)}
        for name in colors:
            rows=record['old_reference'] if name=='old reference' else [r for r in record['production_comparisons'] if r['backend']==name]
            points=[project(r['rtol'],r['max_event_difference_from_quadrature']) for r in rows]
            pd.line(points,fill=colors[name],width=2)
            for x,y in points:pd.ellipse((x-3,y-3,x+3,y+3),fill=colors[name])
        pd.text((80,18),'Seventh crossing: maximum coordinate-component error vs independent quadrature',fill=(15,20,30))
        for i,(name,color) in enumerate(colors.items()):pd.text((85+i*210,46),name,fill=color)
        pd.text((240,496),'Requested rtol (loose to tight); vertical scale: error in M',fill=(20,20,25))
        plot.save(output/'seventh_convergence.png')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=DEFAULT)
    parser.add_argument('--workers',type=int,default=10)
    parser.add_argument('--rays',action='store_true');parser.add_argument('--shadows',action='store_true')
    parser.add_argument('--seventh',action='store_true');parser.add_argument('--images',action='store_true')
    parser.add_argument('--benchmark',action='store_true')
    parser.add_argument('--figures-only',action='store_true',help='draw figures from saved measurements; never traces')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    if args.figures_only:
        if any((args.rays,args.shadows,args.seventh,args.images,args.benchmark)):
            parser.error('--figures-only cannot be combined with tracing/audit flags')
        figures_from_saved(args.output);return
    if not any((args.rays,args.shadows,args.seventh,args.images,args.benchmark)):
        args.rays=args.shadows=args.seventh=True
    if args.rays:ray_audit(args.output)
    if args.seventh:seventh_audit(args.output)
    if args.shadows:shadow_audit(args.output)
    if args.images:small_images(args.output,args.workers)
    if args.benchmark:benchmarks(args.output,args.workers)


if __name__=='__main__':main()
