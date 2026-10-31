"""Kerr disks: invariant physics, independent references, opaque rendering."""
from dataclasses import replace
from types import SimpleNamespace
import json,hashlib
import numpy as np
import pytest
from scipy.integrate import solve_ivp
from numerical_observatory.kerr import Kerr
from numerical_observatory.metric import Schwarzschild
from numerical_observatory.camera import Camera
from numerical_observatory.disk import make_disk
from numerical_observatory.emitter_motion import KerrCircular,kerr_isco,kerr_orbit_sign
from numerical_observatory.kerr_disk_reference import (independent_isco,circular_bl,circular_integrals,
    massive_radial,circular_residual,bl_frequency_shift,separated_hits)
from numerical_observatory.kerr_support import inclined_camera_options,default_domain
from numerical_observatory.kerr_reference import pixel_momentum
from numerical_observatory.geodesic import trace,Integration,hamilton_rhs
from numerical_observatory.emission import frequency_shift,photon_frequency
from numerical_observatory.background import CelestialGrid
from numerical_observatory.render import render


def context(spin=.9,orbit='prograde',inclination=85,width=640,height=480):
    m=Kerr(spin=spin)
    c=Camera(**inclined_camera_options(m,dict(radius=60,inclination_degrees=inclination,
                                             width=width,height=height,fov_degrees=60)))
    s=Integration(rtol=1e-11,atol=1e-13,escape_radius=110,max_affine=400,max_step=2)
    d=make_disk(m,dict(enabled=True,orbit=orbit,display=dict(palette='blackbody',exposure=5)))
    return m,c,s,d


@pytest.mark.parametrize('spin,pro,retro',[(0,6,6),(.5,4.233002529530825,7.554584714512359),
    (.9,2.320883041761887,8.717352279606489),(-.9,2.320883041761887,8.717352279606489),
    (.99,1.454497938059672,8.971861342546969),(-.99,1.454497938059672,8.971861342546969)])
def test_isco_marginal_stability_reference(spin,pro,retro):
    m=Kerr(mass=2,spin=spin)
    for orbit,expected in [('prograde',pro),('retrograde',retro)]:
        r=kerr_isco(m,orbit)
        assert r/2==pytest.approx(expected,abs=2e-13)
        assert independent_isco(m,orbit)==pytest.approx(r,abs=1e-11)
        for factor,sign in [(.999,1),(1.001,-1)]:
            E,L=circular_integrals(m,r*factor,orbit)
            assert np.sign(massive_radial(m,r*factor,E,L)[2])==sign


@pytest.mark.parametrize('spin',[-.99,-.9,0,.5,.9,.99])
@pytest.mark.parametrize('orbit',['prograde','retrograde'])
def test_emitter_normalization_geodesic_and_integrals(spin,orbit):
    m=Kerr(spin=spin); motion=KerrCircular(m,orbit)
    for r in [kerr_isco(m,orbit),12.,20.]:
        for phi in [0,.7,2.4]:
            rho=np.sqrt(r*r+m.a*m.a); event=np.array([0.,rho*np.cos(phi),rho*np.sin(phi),0.])
            u=motion.four_velocity(event); cov=m.covariant(event)@u
            assert u @ cov==pytest.approx(-1,abs=2e-14)
            assert circular_residual(m,motion,event)<4e-13
            omega,ut,E,L=circular_bl(m,r,orbit)
            assert motion.omega(r)==pytest.approx(omega,rel=2e-14)
            np.testing.assert_allclose([u[0],-cov[0],event[1]*cov[2]-event[2]*cov[1]],
                                       [ut,E,L],atol=5e-13,rtol=2e-14)
            np.testing.assert_allclose(circular_integrals(m,r,orbit),[E,L],atol=3e-13)
            np.testing.assert_allclose(m.bl_jacobian(event) @ np.array([ut,0,0,omega*ut]),u,atol=2e-14)
    assert motion.rotation==kerr_orbit_sign(m,orbit)


@pytest.mark.parametrize('orbit',['prograde','retrograde'])
def test_timelike_orbit_integration(orbit):
    m=Kerr(spin=.9); motion=KerrCircular(m,orbit); r=max(8.,kerr_isco(m,orbit)*1.2)
    e=np.array([0.,np.sqrt(r*r+m.a*m.a),0.,0.]); u=motion.four_velocity(e); p=m.covariant(e)@u
    period=2*np.pi/abs(motion.omega(r)*u[0])
    sol=solve_ivp(lambda t,y:hamilton_rhs(m,y),(0,2*period),np.r_[e,p],method='DOP853',
                  rtol=1e-11,atol=1e-13,max_step=period/80)
    assert sol.success
    radii=np.array([m.radius(x) for x in sol.y[:4].T])
    np.testing.assert_allclose(radii,r,atol=3e-8,rtol=0)
    np.testing.assert_allclose(sol.y[3],0,atol=1e-12)
    np.testing.assert_allclose(sol.y[4],p[0],atol=1e-12)


def test_configuration_stability_and_kerr_radius():
    m=Kerr(spin=.9); d=make_disk(m,dict(enabled=True))
    assert d.surface.inner_radius==kerr_isco(m)
    for bad in [dict(inner_radius=2),dict(inner_radius='wrong'),dict(outer_radius=2),
                dict(inclination_degrees=85),dict(rotation=-1),dict(orbit='wrong')]:
        with pytest.raises((ValueError,TypeError)): make_disk(m,dict(enabled=True,**bad))
    for offset,expected in [(-1e-5,False),(1e-5,True)]:
        for radius in (d.surface.inner_radius+offset,20-offset):
            event=np.array([0.,np.sqrt(radius*radius+m.a*m.a),0.,0.])
            assert d.surface.contains(event)==expected
    r=4.; event=np.array([0.,np.sqrt(r*r+m.a*m.a),0.,0.])
    assert d.surface.radius(event)==pytest.approx(r)
    assert d.emission.temperature(event)==pytest.approx(1e4*(r/d.surface.inner_radius)**(-.75))
    assert np.linalg.norm(event[1:])!=pytest.approx(r)
    assert KerrCircular(Kerr()).rotation==1
    assert KerrCircular(Kerr(),'retrograde').rotation==-1
    assert KerrCircular(Kerr(spin=-.9)).rotation==-1


def test_all_plane_crossings_not_emitting_ordinals_and_no_duplicates():
    m,_,_,d=context(orbit='retrograde')
    def dense(t):
        t=np.asarray(t); r=np.interp(t,[1,2,3,4],[4.,7.,22.,12.])
        z=(t-1)*(t-2)*(t-3)*(t-4)
        return np.array([t,np.sqrt(r*r+m.a*m.a),np.zeros_like(t),z,
                         np.ones_like(t),np.zeros_like(t),np.zeros_like(t),np.zeros_like(t)])
    sol=SimpleNamespace(t=np.linspace(0,5,21),sol=dense)
    hits=d.surface.crossings(sol)
    assert [h.crossing_index for h in hits]==[1,2,3,4]
    np.testing.assert_allclose([h.affine for h in hits],[1,2,3,4],atol=1e-11)
    assert [d.surface.contains(h.event) for h in hits]==[False,False,False,True]
    assert d.surface.first_hit(sol).crossing_index==4


# Explicit camera subpixel positions: all are actual Hamiltonian trajectories.
HIT_CASES=[(1,(382.05182957538767,247.04245815155477)),
           (2,(304.4455154940234,204.24704476156725)),
           (3,(367.0613895063265,275.17830073795415)),
           (4,(339.69077568795615,286.7805635605909))]


@pytest.mark.parametrize('ordinal,xy',HIT_CASES)
def test_independent_intersections_redshift_and_opacity(ordinal,xy):
    m,c,s,d=context(orbit='retrograde'); p=pixel_momentum(m,c,*xy)
    result,sol=trace(m,c.event,p,s,surface=d.surface,return_solution=True)
    assert result.status=='emitted' and result.hit.crossing_index==ordinal
    assert sol.t[-1]==result.hit.affine
    with pytest.raises(ValueError): sol.sol(result.hit.affine+.1)
    independent=separated_hits(m,c,p,result.hit.affine+.05)
    valid=[h for h in independent if d.surface.contains(h.event)]
    assert valid and valid[0].crossing_index==ordinal
    np.testing.assert_allclose(result.hit.event,valid[0].event,atol=2e-5,rtol=0)
    np.testing.assert_allclose(result.hit.momentum,valid[0].momentum,atol=2e-6,rtol=0)
    g=frequency_shift(p,c.tetrad(m)[0],result.hit.momentum,d.motion.four_velocity(result.hit.event))
    assert g==pytest.approx(bl_frequency_shift(m,c,p,m.radius(result.hit.event),d.motion.orbit),rel=2e-8)
    assert d.emission.observed_bolometric(result.hit.event,g)==pytest.approx(g**4*d.emission.bolometric(result.hit.event),rel=2e-15)
    assert result.max_null_error<s.null_tolerance


@pytest.mark.parametrize('xy',[(270.,247.5),(370.,247.5)])
def test_prograde_approaching_receding_frequency_and_capture_after_hit(xy):
    m,c,s,d=context(); p=pixel_momentum(m,c,*xy)
    result=trace(m,c.event,p,s,surface=d.surface)
    assert result.status=='emitted'
    h=result.hit; u=d.motion.four_velocity(h.event)
    assert photon_frequency(p,c.tetrad(m)[0])>0 and photon_frequency(h.momentum,u)>0
    g=frequency_shift(p,c.tetrad(m)[0],h.momentum,u)
    assert g==pytest.approx(bl_frequency_shift(m,c,p,m.radius(h.event)),rel=2e-9)
    with pytest.raises(ValueError): photon_frequency(-h.momentum,u)
    if xy[0]<320: assert u[1]>0  # moving toward observer's +X side
    else: assert u[1]<0


def test_captured_after_first_hit_and_unresolved_near_tangent():
    m,c,s,d=context(); p=pixel_momentum(m,c,327.5,247.5)
    assert trace(m,c.event,p,s).status=='captured'
    assert trace(m,c.event,p,s,surface=d.surface).status=='emitted'
    assert trace(m,c.event,p,replace(s,max_affine=.1),surface=d.surface).status=='unresolved'
    assert trace(m,c.event,p,replace(s,null_tolerance=1e-17),surface=d.surface).status=='unresolved'
    m,c,s,d=context(inclination=90)
    p=pixel_momentum(m,c,420,239.5)
    assert trace(m,c.event,p,s,surface=d.surface).reason=='coplanar_surface'
    c=replace(c,position=(60.,0.,1e-4),target=(0,0,0))
    p=pixel_momentum(m,c,420,239.5)
    assert trace(m,c.event,p,s,surface=d.surface).reason!='coplanar_surface'
    # Stable high-spin material can lie inside the ergosphere. A circular
    # emitter remains timelike there; substituting a static velocity would fail.
    m,c,s,d=context(spin=.99); p=pixel_momentum(m,c,319.5,241.)
    result=trace(m,c.event,p,s,surface=d.surface)
    assert result.status=='emitted' and kerr_isco(m)<m.radius(result.hit.event)<2*m.mass
    u=d.motion.four_velocity(result.hit.event)
    assert u @ m.covariant(result.hit.event) @ u==pytest.approx(-1,abs=2e-14)
    g=frequency_shift(p,c.tetrad(m)[0],result.hit.momentum,u)
    assert g==pytest.approx(bl_frequency_shift(m,c,p,m.radius(result.hit.event)),rel=1e-9)


def test_zero_spin_disk_rotated_schwarzschild_equivalence():
    m,c,s,d=context(spin=0); sm=Schwarzschild(); inclination=np.deg2rad(85)
    sine,cosine=np.sin(inclination),np.cos(inclination)
    rotation=np.array([[0.,-cosine,sine],[1.,0.,0.],[0.,sine,cosine]])
    sc=Camera(position=(0,0,60),width=640,height=480)
    sd=make_disk(sm,dict(enabled=True,inclination_degrees=85,display=dict(palette='blackbody',exposure=5)))
    statuses=set(); indices=[]
    for x,y in [(420,232),(310,192),(351,276),(287,274),(319,200),(0,0)]:
        kp=pixel_momentum(m,c,x,y); sp=pixel_momentum(sm,sc,x,y)
        np.testing.assert_allclose(kp,np.r_[sp[0],rotation@sp[1:]],atol=1e-15)
        a=trace(m,c.event,kp,s,surface=d.surface); b=trace(sm,sc.event,sp,s,surface=sd.surface)
        assert a.status==b.status; statuses.add(a.status)
        if a.hit:
            indices.append(a.hit.crossing_index)
            assert a.hit.crossing_index==b.hit.crossing_index
            np.testing.assert_allclose(a.hit.event[1:],rotation@b.hit.event[1:],atol=3e-6,rtol=0)
            ua=d.motion.four_velocity(a.hit.event); ub=sd.motion.four_velocity(b.hit.event)
            np.testing.assert_allclose(ua,np.r_[ub[0],rotation@ub[1:]],atol=2e-7)
            ga=frequency_shift(kp,c.tetrad(m)[0],a.hit.momentum,ua)
            gb=frequency_shift(sp,sc.tetrad(sm)[0],b.hit.momentum,ub)
            assert ga==pytest.approx(gb,rel=3e-7)
            Ia=d.emission.observed_bolometric(a.hit.event,ga); Ib=sd.emission.observed_bolometric(b.hit.event,gb)
            assert Ia==pytest.approx(Ib,rel=2e-6)
            np.testing.assert_allclose(d.display.linear_rgb(Ia,ga*d.emission.temperature(a.hit.event)),
                sd.display.linear_rgb(Ib,gb*sd.emission.temperature(b.hit.event)),atol=3e-7)
        elif a.direction is not None:
            np.testing.assert_allclose(a.direction,rotation@b.direction,atol=1e-8)
    assert max(indices)>=4 and {'captured','escaped','emitted'}<=statuses


def test_saved_data_spawn_supersampling_repaint_and_diagnostics(tmp_path,monkeypatch):
    from numerical_observatory.surface_render import repaint_surface
    from numerical_observatory.disk_diagnostics import generate,extract
    m,c,s,d=context(width=6,height=4,inclination=60)
    for workers in (1,2):
        report=render(m,c,s,tmp_path/str(workers),disk=d,background=CelestialGrid(),workers=workers,samples=2,method='quadrature')
        assert not report['exact_rotational_path_reuse'] and not report['quadrature_used']
        assert report['spacetime']['spin']==.9 and report['orbit']=='prograde'
        assert report['counts']['emitted']>0 and report['counts']['unresolved']==0
    with np.load(tmp_path/'1/disk_rays.npz') as a,np.load(tmp_path/'2/disk_rays.npz') as b:
        for key in a.files: np.testing.assert_array_equal(a[key],b[key])
        emitted=a['status']==3
        np.testing.assert_allclose(a['bolometric_intensity'][emitted],a['g'][emitted]**4*a['emitted_bolometric_intensity'][emitted])
        raw={k:a[k] for k in a.files}
    metadata=json.loads((tmp_path/'1/diagnostics.json').read_text())
    arrays,details=extract(raw,metadata)
    assert details['max_saved_radius_position_disagreement']<1e-12
    assert not details['warnings']
    fallback=raw.copy(); del fallback['hit_radius']
    again,_=extract(fallback,metadata)
    np.testing.assert_allclose(again['radius_m'],arrays['radius_m'])
    raw_hash=hashlib.sha256((tmp_path/'1/disk_rays.npz').read_bytes()).hexdigest()
    def no_trace(*a,**kw): pytest.fail('postprocessing retraced photons')
    import numerical_observatory.surface_render as module
    monkeypatch.setattr(module,'trace',no_trace)
    for palette in ('blackbody','thermal','gray'):
        repaint_surface(tmp_path/'1',tmp_path/palette,display=dict(palette=palette))
    assert (tmp_path/'1/shadow.png').read_bytes()==(tmp_path/'blackbody/shadow.png').read_bytes()
    assert hashlib.sha256((tmp_path/'1/disk_rays.npz').read_bytes()).hexdigest()==raw_hash
    record=generate(tmp_path/'1',tmp_path/'maps',bounds=(1,1,5,3))
    assert not record['geodesics_retraced'] and record['spacetime']['type']=='kerr'
    from numerical_observatory.orbital_diagnostics import compute_angles
    from numerical_observatory.disk_layers import collect,scene_from_metadata
    for call in (lambda:compute_angles(metadata,raw),lambda:collect(m,c,s,d),lambda:scene_from_metadata(metadata)):
        with pytest.raises(ValueError,match='Kerr'): call()


def test_spin_reversal_disk_mapping():
    m,c,s,d=context(width=8,height=6); n,cn,_,dn=context(spin=-.9,width=8,height=6)
    for x,y in [(1,3),(2,3),(4,3),(6,3)]:
        a=trace(m,c.event,pixel_momentum(m,c,x,y),s,surface=d.surface)
        b=trace(n,cn.event,pixel_momentum(n,cn,7-x,y),s,surface=dn.surface)
        assert a.status==b.status
        if a.hit:
            np.testing.assert_allclose(a.hit.event,b.hit.event*np.array([1,1,-1,1]),atol=1e-7)
            assert a.hit.crossing_index==b.hit.crossing_index
            ga=frequency_shift(pixel_momentum(m,c,x,y),c.tetrad(m)[0],a.hit.momentum,d.motion.four_velocity(a.hit.event))
            gb=frequency_shift(pixel_momentum(n,cn,7-x,y),cn.tetrad(n)[0],b.hit.momentum,dn.motion.four_velocity(b.hit.event))
            assert ga==pytest.approx(gb,rel=1e-8)


def test_kerr_prefix_audit_ignores_unphysical_later_geometry():
    from numerical_observatory.geodesic import summarize_solution
    from numerical_observatory.kerr_support import finish_kerr
    m,c,s,d=context(); p=pixel_momentum(m,c,327.5,247.5)
    domain=default_domain(m,c.event,s)
    original,sol=trace(m,c.event,p,s,domain=domain,return_solution=True)
    first=d.surface.first_hit(sol)
    assert first is not None
    def corrupted(t):
        y=sol.sol(t).copy()
        y[4]+=np.where(np.asarray(t)>first.affine+.01,.2,0.)
        return y
    altered=SimpleNamespace(t=sol.t,y=sol.y,sol=corrupted,success=sol.success,
        message=sol.message,nfev=sol.nfev,t_events=sol.t_events)
    result=summarize_solution(m,c.event,p,s,altered,domain=domain,builtin_domain=False,surface=d.surface)
    result=finish_kerr(m,c.event,p,s,altered,result,False)
    assert result.status=='emitted' and result.max_null_error<s.null_tolerance
    without=summarize_solution(m,c.event,p,s,altered,domain=domain,builtin_domain=False)
    assert without.status=='unresolved'


def test_static_camera_reference_and_kerr_zoom(tmp_path):
    from numerical_observatory.diagnostic_zoom import trace_window
    m,c,s,d=context(width=6,height=4,inclination=60)
    c=replace(c,observer='static')
    p=pixel_momentum(m,c,2,2); result=trace(m,c.event,p,s,surface=d.surface)
    assert result.status=='emitted'
    h=result.hit; g=frequency_shift(p,c.tetrad(m)[0],h.momentum,d.motion.four_velocity(h.event))
    assert g==pytest.approx(bl_frequency_shift(m,c,p,m.radius(h.event)),rel=1e-9)
    render(m,c,s,tmp_path/'full',disk=d,background=CelestialGrid())
    trace_window(tmp_path/'full',tmp_path/'zoom',(1,1,4,3),1)
    with np.load(tmp_path/'full/disk_rays.npz') as a,np.load(tmp_path/'zoom/disk_rays.npz') as b:
        for key in ('status','g','hit_radius','hit_position','crossing_index','bolometric_intensity'):
            np.testing.assert_allclose(a[key][1:3,1:4],b[key],rtol=0,atol=0)


def test_small_spin_isco_avoids_cancellation():
    for spin in (1e-8,-1e-8,1e-6,-1e-6):
        m=Kerr(spin=spin)
        for orbit in ('prograde','retrograde'):
            assert kerr_isco(m,orbit)==pytest.approx(independent_isco(m,orbit),abs=1e-11)
        assert kerr_isco(m,'prograde')<6<kerr_isco(m,'retrograde')
