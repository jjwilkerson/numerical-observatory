"""Independent checks for selectable compact Carter Kerr tracing."""
from dataclasses import replace
import json
import numpy as np
import pytest
from numerical_observatory.cli import load_configuration
from numerical_observatory.kerr import Kerr
from numerical_observatory.metric import Schwarzschild
from numerical_observatory.camera import Camera
from numerical_observatory.kerr_support import inclined_camera_options
from numerical_observatory.kerr_reference import pixel_momentum,shadow_curve,reference_radius
from numerical_observatory.kerr_carter import CarterOrbit,CarterFailure
from numerical_observatory.kerr_carter_quadrature import escaping_crossing
from numerical_observatory.geodesic import Integration,trace
from numerical_observatory.emission import frequency_shift
from numerical_observatory.disk import make_disk
from numerical_observatory.background import CelestialGrid,Starfield
from numerical_observatory.render import render


def scene(spin=.9,inc=85,width=640,height=480):
    m=Kerr(spin=spin)
    c=Camera(**inclined_camera_options(m,dict(radius=60,inclination_degrees=inc,width=width,height=height,fov_degrees=60)))
    s=Integration(rtol=1e-11,atol=1e-13,escape_radius=110,max_affine=400,max_step=4)
    d=make_disk(m,dict(enabled=True,display=dict(palette='blackbody',exposure=5)))
    return m,c,s,d


def test_configuration_backend_and_invalid_combinations(tmp_path):
    p=tmp_path/'cfg.toml'
    p.write_text('[spacetime]\ntype="kerr"\nspin=.9\n[geodesics]\nintegrator="carter"\nfallback="hamiltonian"\n'.replace('spin=.9','spin=0.9'))
    m,c,s=load_configuration(p)
    assert s.integrator=='carter' and s.carter_fallback=='hamiltonian'
    assert Integration().integrator=='hamiltonian'
    with pytest.raises(ValueError):Integration(integrator='wrong')
    with pytest.raises(ValueError):Integration(carter_fallback='wrong')
    p.write_text('[geodesics]\nintegrator="carter"\n')
    with pytest.raises(ValueError,match='exact Kerr'):load_configuration(p)
    with pytest.raises(ValueError,match='exact Kerr'):
        trace(Schwarzschild(),c.event,c.momentum(m,[.4,0,-1]),s)
    with pytest.raises(ValueError,match='exact Kerr'):
        render(Schwarzschild(),c,s,tmp_path/'invalid',method='quadrature')


@pytest.mark.parametrize('spin,inc',[(0,90),(.5,20),(.9,60),(-.9,85),(.99,85),(.9,90)])
def test_common_initial_data_positions_momenta_and_metric_residuals(spin,inc):
    m,c,s,d=scene(spin,inc)
    for xy in ((0,0),(270,247.5),(319,200),(370,247.5)):
        p=pixel_momentum(m,c,*xy)
        h,hs=trace(m,c.event,p,s,surface=d.surface,return_solution=True)
        a,cs=trace(m,c.event,p,replace(s,integrator='carter'),surface=d.surface,return_solution=True)
        assert a.status==h.status
        assert a.fallback_reason is None and a.backend=='carter'
        np.testing.assert_array_equal(cs.sol(0),np.r_[c.event,p])
        for t in np.linspace(0,min(cs.t[-1],hs.t[-1])*.999,11):
            state=cs.sol(t); reference=hs.sol(t)
            np.testing.assert_allclose(state[:4],reference[:4],atol=3e-6,rtol=0)
            np.testing.assert_allclose(state[4:],reference[4:],atol=5e-7,rtol=0)
            y=cs.separated_solution.sol(t/m.mass)
            tangent=cs.carter_orbit.tangent(y)
            np.testing.assert_allclose(tangent,m.inverse(state[:4])@state[4:],atol=2e-7,rtol=0)
        assert a.max_null_error<s.null_tolerance
        assert a.carter_drift<s.null_tolerance
        if a.hit:
            assert a.hit.crossing_index==h.hit.crossing_index
            ga=frequency_shift(p,c.tetrad(m)[0],a.hit.momentum,d.motion.four_velocity(a.hit.event))
            gh=frequency_shift(p,c.tetrad(m)[0],h.hit.momentum,d.motion.four_velocity(h.hit.event))
            assert ga==pytest.approx(gh,rel=2e-7)
            assert d.emission.observed_bolometric(a.hit.event,ga)==pytest.approx(d.emission.observed_bolometric(h.hit.event,gh),rel=2e-6)
        elif a.direction is not None:
            np.testing.assert_allclose(a.direction,h.direction,atol=2e-7,rtol=0)


def test_mass_scaling_round_trip_and_turning_points():
    m=Kerr(2,.9)
    c=Camera(**inclined_camera_options(Kerr(spin=.9),dict(radius=30,inclination_degrees=60,width=64,height=48,fov_degrees=35)))
    c=replace(c,position=tuple(2*np.asarray(c.position)))
    s=Integration(rtol=1e-11,atol=1e-13,escape_radius=140,max_affine=700,max_step=4,integrator='carter',kerr_source_radius=2e5)
    p=pixel_momentum(m,c,56,30)
    result,sol=trace(m,c.event,p,s,return_solution=True)
    assert result.status=='escaped'
    orbit=sol.carter_orbit
    np.testing.assert_allclose(orbit.cartesian(orbit.initial),np.r_[c.event,p],atol=3e-14)
    ys=sol.separated_solution.sol(np.linspace(0,sol.separated_solution.t[-1],501))
    assert ys[1].min()<0<ys[1].max()
    assert ys[3].min()<0<ys[3].max()
    radial=orbit.M/ys[0]
    assert radial.min()<m.radius(c.event) and radial[-1]==pytest.approx(140,abs=1e-8)


def test_first_four_and_seventh_crossings_independent_quadrature():
    m,c,s,d=scene();d=make_disk(m,dict(enabled=True,orbit='retrograde'))
    for ordinal,xy in [(1,(382.05182957538767,247.04245815155477)),
        (2,(304.4455154940234,204.24704476156725)),(3,(367.0613895063265,275.17830073795415)),
        (4,(339.69077568795615,286.7805635605909)),(7,(314.5056064910384,280.9199781195458))]:
        p=pixel_momentum(m,c,*xy)
        q,info=escaping_crossing(m,c.event,p,ordinal)
        a,sol=trace(m,c.event,p,replace(s,integrator='carter'),surface=d.surface,return_solution=True)
        h=trace(m,c.event,p,s,surface=d.surface)
        assert a.status==h.status=='emitted'
        assert a.hit.crossing_index==h.hit.crossing_index==ordinal
        # Unchanged original extreme-case agreement threshold.
        np.testing.assert_allclose(a.hit.event,q.event,atol=3e-4,rtol=0)
        np.testing.assert_allclose(h.hit.event,q.event,atol=3e-4,rtol=0)
        np.testing.assert_allclose(a.hit.momentum,q.momentum,atol=1e-5,rtol=0)
        assert sol.t[-1]==a.hit.affine
        assert all(np.all(x<=a.hit.affine) for x in sol.t_events)
        with pytest.raises(ValueError):sol.sol(a.hit.affine+.01)
        if ordinal==7:
            assert abs(sol.separated_solution.sol(a.hit.affine/m.mass)[4])>2*np.pi
            assert info['polar_phase']>6*np.pi
            q2,_=escaping_crossing(m,c.event,p,ordinal,precision=70,eps=5e-13)
            np.testing.assert_allclose(q.event,q2.event,atol=1e-7,rtol=0)


def test_tail_convergence_and_near_axis_error_control():
    m,c,s=load_configuration('examples/kerr/spin09_i20.toml')
    c=replace(c,width=16,height=12);p=pixel_momentum(m,c,8,0)
    a=trace(m,c.event,p,replace(s,integrator='carter'))
    h=trace(m,c.event,p,s)
    assert a.status==h.status=='escaped' and a.carter_drift<s.null_tolerance
    np.testing.assert_allclose(a.direction,h.direction,atol=3e-7,rtol=0)
    m,c,s,_=scene();p=pixel_momentum(m,c,0,0)
    results=[trace(m,c.event,p,replace(s,integrator='carter',kerr_source_radius=R)) for R in (1e4,1e5,1e6)]
    d1=np.linalg.norm(results[0].direction-results[1].direction)
    d2=np.linalg.norm(results[1].direction-results[2].direction)
    assert d2<d1/50


def test_explicit_axis_fallback_and_failures(monkeypatch):
    m,c,s,_=scene(inc=0);p=pixel_momentum(m,c,350,200)
    result=trace(m,c.event,p,replace(s,integrator='carter'))
    assert result.status=='unresolved' and 'polar_axis' in result.reason
    a=trace(m,c.event,p,replace(s,integrator='carter',carter_fallback='hamiltonian'))
    h=trace(m,c.event,p,s)
    assert a.backend=='hamiltonian' and a.fallback_reason=='polar_axis_initial_condition'
    assert a.status==h.status
    if a.direction is not None:np.testing.assert_array_equal(a.direction,h.direction)
    m,c,s,_=scene();p=pixel_momentum(m,c,0,0)
    def broken(*args):raise CarterFailure('deliberate_test_failure')
    monkeypatch.setattr(CarterOrbit,'mino_rhs',broken)
    a=trace(m,c.event,p,replace(s,integrator='carter'))
    assert a.status=='unresolved' and 'deliberate' in a.reason and a.evaluations>0
    a=trace(m,c.event,p,replace(s,integrator='carter',carter_fallback='hamiltonian'))
    assert a.status=='escaped' and a.fallback_reason=='deliberate_test_failure'


def test_affine_limit_coplanar_and_exact_critical_not_forced():
    m,c,s,d=scene(inc=90);p=pixel_momentum(m,c,420,239.5)
    a=trace(m,c.event,p,replace(s,integrator='carter'),surface=d.surface)
    assert a.status=='unresolved' and a.reason=='coplanar_surface'
    a=trace(m,c.event,p,replace(s,integrator='carter',max_affine=.01))
    assert a.status=='unresolved' and a.direction is None
    m,c,s,_=scene(spin=0,inc=90);q=shadow_curve(m,c,401)[0]
    a=trace(m,c.event,pixel_momentum(m,c,*q),replace(s,integrator='carter',carter_fallback='hamiltonian'))
    assert a.status=='unresolved' and a.direction is None


def test_spawn_supersampling_data_and_image_equivalence(tmp_path):
    m,c,s,d=scene(width=8,height=6,inc=60)
    s=replace(s,rtol=1e-9,atol=1e-11,max_step=4)
    for workers in (1,2,3):
        report=render(m,c,replace(s,integrator='carter'),tmp_path/str(workers),disk=d,background=CelestialGrid(),workers=workers,samples=2)
        assert report['integrator']=='carter' and report['fallback_count']==0
        assert not report['exact_rotational_path_reuse'] and report['counts']['unresolved']==0
        assert report['carter_accepted_rays']==8*6*4
    with np.load(tmp_path/'1/disk_rays.npz') as a:
        for workers in (2,3):
            with np.load(tmp_path/str(workers)/'disk_rays.npz') as b:
                for key in a.files:np.testing.assert_array_equal(a[key],b[key])
        assert np.all(a['backend']==1) and not np.any(a['backend_failure_code'])
    render(m,c,s,tmp_path/'ham',disk=d,background=CelestialGrid(),samples=2)
    with np.load(tmp_path/'1/disk_rays.npz') as a,np.load(tmp_path/'ham/disk_rays.npz') as b:
        np.testing.assert_array_equal(a['status'],b['status'])
        np.testing.assert_array_equal(a['crossing_index'],b['crossing_index'])
        emitted=a['status']==3
        np.testing.assert_allclose(a['g'][emitted],b['g'][emitted],atol=2e-6,rtol=0)
        np.testing.assert_allclose(a['hit_position'][emitted],b['hit_position'][emitted],atol=1e-4,rtol=0)
        np.testing.assert_allclose(a['bolometric_intensity'][emitted],b['bolometric_intensity'][emitted],rtol=2e-5)
    for palette in ('thermal','gray','blackbody'):
        from numerical_observatory.surface_render import repaint_surface
        repaint_surface(tmp_path/'1',tmp_path/palette,display={'palette':palette})


def test_fallback_saved_provenance(tmp_path):
    m,c,s,d=scene(inc=0,width=3,height=3)
    s=replace(s,integrator='carter',carter_fallback='hamiltonian')
    report=render(m,c,s,tmp_path,background=Starfield())
    assert report['fallback_count']==9 and report['carter_accepted_rays']==0
    with np.load(tmp_path/'rays.npz') as z:
        assert np.all(z['backend']==0) and np.all(z['backend_failure_code']>0)
    assert report['carter_failure_reasons']=={'polar_axis_initial_condition':9}


def test_outward_initial_direction_and_equation_residual():
    from numerical_observatory.geodesic import hamilton_rhs
    m,c,s,d=scene();c=replace(c,target=tuple(2*np.asarray(c.position)))
    p=pixel_momentum(m,c,340,220)
    h=trace(m,c.event,p,s)
    a,sol=trace(m,c.event,p,replace(s,integrator='carter'),return_solution=True)
    assert a.status==h.status=='escaped' and sol.carter_orbit.initial[1]<0
    np.testing.assert_allclose(a.direction,h.direction,atol=1e-7,rtol=0)
    orbit=sol.carter_orbit
    for t in np.linspace(0,sol.t[-1]/m.mass,9):
        y=sol.separated_solution.sol(t); rate=orbit.rhs(0,y)
        step=1e-4
        numerical=(-orbit.cartesian(y+2*step*rate)+8*orbit.cartesian(y+step*rate)
                   -8*orbit.cartesian(y-step*rate)+orbit.cartesian(y-2*step*rate))/(12*step*m.mass)
        physical=hamilton_rhs(m,orbit.cartesian(y))
        np.testing.assert_allclose(numerical,physical,atol=2e-7,rtol=0)


@pytest.mark.parametrize('spin,inc',[(0,90),(.9,20),(.99,85)])
def test_numerical_finite_shadow_brackets(spin,inc):
    from numerical_observatory.carter_validation import boundary_sample
    m,c,s,d=scene(spin,inc,width=96,height=72)
    curve=shadow_curve(m,c,10001);angle=.31;radius=reference_radius(curve,c,angle)
    results=[]
    for backend in ('hamiltonian','carter'):
        item=boundary_sample(m,c,replace(s,integrator=backend),angle,radius,precision=1e-3)
        assert item['resolved'] and item['bracket_width_pixels']<=1e-3
        assert abs(item['error_pixels'])<1e-3
        results.append(item)
    assert abs(results[0]['radius_pixels']-results[1]['radius_pixels'])<1e-3
