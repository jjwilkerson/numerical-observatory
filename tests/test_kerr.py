"""Kerr physics tests use metric-free Carter equations and finite-camera theory."""
from dataclasses import replace
import json
import numpy as np
import pytest
from numerical_observatory.kerr import Kerr
from numerical_observatory.metric import ETA, Schwarzschild
from numerical_observatory.camera import Camera
from numerical_observatory.geodesic import Integration, trace
from numerical_observatory.kerr_support import inclined_camera_options
from numerical_observatory.kerr_reference import (shadow_curve,pixel_momentum,separated_reference,potentials)


def camera(metric, inclination=60, **kw):
    return Camera(**inclined_camera_options(metric,dict(radius=30,inclination_degrees=inclination,
                                                        width=64,height=48,fov_degrees=35,**kw)))


def settings(**kw):
    return Integration(escape_radius=70,max_affine=350,rtol=1e-10,atol=1e-12,**kw)


@pytest.mark.parametrize('spin',[-.99,-.9,0.,.5,.99])
def test_metric_inverse_derivatives_and_bl_form(spin):
    m=Kerr(2,spin)
    h=m.horizon_radius*1.002
    near=np.array([0.,np.sqrt(h*h+m.a*m.a)*np.sin(.7),0.,h*np.cos(.7)])
    for x in (np.array([1.,8.,3.,5.]),np.array([0.,0.,0.,8.]),np.array([0.,1e-7,0.,5.]),near):
        r,ell,f=m.geometry(x)
        assert abs(ell @ ETA @ ell)<2e-15
        np.testing.assert_allclose(m.covariant(x) @ m.inverse(x),np.eye(4),atol=8e-15)
        assert np.linalg.det(m.covariant(x))==pytest.approx(-1,abs=4e-15)
        d=m.inverse_derivative(x)
        assert not d[0].any()
        for i in range(1,4):
            v=np.eye(4)[i]*2e-4
            fd=(-m.inverse(x+2*v)+8*m.inverse(x+v)-8*m.inverse(x-v)+m.inverse(x-2*v))/(12*2e-4)
            np.testing.assert_allclose(d[i],fd,atol=3e-11,rtol=2e-9)
    x=np.array([0.,7.,2.,4.]); r=m.radius(x); c=x[3]/r; s2=1-c*c
    sigma=r*r+m.a*m.a*c*c; delta=r*r-2*m.mass*r+m.a*m.a
    gbl=np.zeros((4,4)); gbl[0,0]=-(1-2*m.mass*r/sigma)
    gbl[0,3]=gbl[3,0]=-2*m.mass*m.a*r*s2/sigma
    gbl[1,1]=sigma/delta; gbl[2,2]=sigma
    gbl[3,3]=(r*r+m.a*m.a+2*m.mass*m.a*m.a*r*s2/sigma)*s2
    jac=m.bl_jacobian(x)
    np.testing.assert_allclose(jac.T @ m.covariant(x) @ jac,gbl,atol=5e-14)


@pytest.mark.parametrize('spin',[-.99,0,.99])
@pytest.mark.parametrize('inclination',[0,20,60,90,180])
def test_horizon_zamo_tetrad_and_initial_nullness(spin,inclination):
    m=Kerr(spin=spin); c=camera(m,inclination)
    frame=c.tetrad(m); g=m.covariant(c.event)
    np.testing.assert_allclose(frame @ g @ frame.T,ETA,atol=3e-15)
    assert (g @ frame[0]) @ np.array([0.,-c.position[1],c.position[0],0.])==pytest.approx(0,abs=2e-15)
    for p in (pixel_momentum(m,c,0,0),pixel_momentum(m,c,31.5,23.5)):
        assert p @ m.inverse(c.event) @ p==pytest.approx(0,abs=3e-15)
        assert (m.inverse(c.event) @ p)[0]<0
        assert p @ frame[0]==pytest.approx(1,abs=2e-15)
    h=m.horizon_radius; theta=.7
    event=np.array([0.,np.sqrt(h*h+m.a*m.a)*np.sin(theta),0.,h*np.cos(theta)])
    assert m.radius(event)==pytest.approx(h,abs=1e-15)
    dr=np.array([0.,h*event[1]/(h*h+m.a*m.a),0.,np.cos(theta)])
    # The radial level-set normal at the horizon is null.
    step=1e-5
    grad=np.r_[0.,[(m.radius(event+np.eye(4)[i]*step)-m.radius(event-np.eye(4)[i]*step))/(2*step) for i in range(1,4)]]
    assert grad @ m.inverse(event) @ grad==pytest.approx(0,abs=2e-10)


def test_ergoregion_and_invalid_inputs():
    m=Kerr(spin=.9); c=Camera(position=(2.,0.,0.),observer='zamo')
    np.testing.assert_allclose(c.tetrad(m)@m.covariant(c.event)@c.tetrad(m).T,ETA,atol=1e-13)
    with pytest.raises(ValueError,match='not timelike'):
        replace(c,observer='static').tetrad(m)
    for spin in [-1,1,float('nan'),True]:
        with pytest.raises(ValueError): Kerr(spin=spin)
    with pytest.raises(ValueError): Kerr(mass=-1)
    with pytest.raises(ValueError): Integration(kerr_capture_fraction=1)
    with pytest.raises(ValueError,match='horizon'):
        Camera(position=(1,0,0),observer='zamo').tetrad(m)


@pytest.mark.parametrize('spin',[-.9,.5,.9])
def test_independent_separated_trajectories_turning_points_and_conservation(spin):
    m=Kerr(spin=spin); c=camera(m)
    p=pixel_momentum(m,c,56,30)
    result,sol=trace(m,c.event,p,settings(),return_solution=True)
    assert result.status=='escaped'
    ref=separated_reference(m,np.r_[c.event,p],sol.t[-1])
    assert ref.success
    times=np.linspace(0,sol.t[-1],501); states=sol.sol(times).T; b=ref.sol(times).T
    radii=np.array([m.radius(y[:4]) for y in states]); theta=np.arccos(states[:,3]/radii)
    np.testing.assert_allclose(radii,b[:,0],atol=2e-7,rtol=0)
    np.testing.assert_allclose(theta,b[:,2],atol=2e-8,rtol=0)
    np.testing.assert_allclose(np.angle(np.exp(1j*(np.arctan2(states[:,2],states[:,1])-b[:,4]))),0,atol=2e-8)
    np.testing.assert_allclose(states[:,0],b[:,5],atol=2e-7,rtol=0)
    assert np.min(b[:,1])<0<np.max(b[:,1])
    assert np.min(b[:,3])<0<np.max(b[:,3])
    assert result.max_null_error<1e-7
    assert result.energy_drift<1e-13
    assert result.angular_momentum_drift<1e-7
    assert result.carter_drift<2e-6


def test_zero_spin_metric_trajectory_camera_and_sky_limit():
    k=Kerr(); s=Schwarzschild(); c=camera(k)
    for x in (c.event,np.array([0.,2.,1.,3.])):
        for name in ('covariant','inverse','inverse_derivative'):
            np.testing.assert_array_equal(getattr(k,name)(x),getattr(s,name)(x))
    np.testing.assert_allclose(c.tetrad(k),replace(c,observer='static').tetrad(s),atol=4e-16)
    for x,y in ((31.5,23.5),(56,30),(49,10)):
        p=pixel_momentum(k,c,x,y)
        a,sa=trace(k,c.event,p,settings(),return_solution=True)
        b,sb=trace(s,c.event,p,settings(),return_solution=True)
        assert a.status==b.status
        np.testing.assert_allclose(sa.y,sb.y,atol=2e-9,rtol=2e-9)
        if a.direction is not None:
            np.testing.assert_allclose(a.direction,b.direction,atol=3e-8,rtol=0)
    base=trace(k,c.event,pixel_momentum(k,c,56,30),settings()).direction
    tiny=Kerr(spin=1e-5)
    d=trace(tiny,c.event,pixel_momentum(tiny,c,56,30),settings()).direction
    assert np.linalg.norm(d-base)<1e-5


def test_shadow_reference_projection_capture_and_spin_reversal():
    for spin in (0,.5,.9,-.9):
        m=Kerr(spin=spin); c=camera(m,90)
        curve=shadow_curve(m,c,401); center=np.array([(c.width-1)/2,(c.height-1)/2])
        if spin==0:
            rad=np.linalg.norm(curve-center,axis=1)
            expected=c.width/(2*np.tan(np.deg2rad(c.fov_degrees)/2))*np.tan(Schwarzschild().shadow_angle(30))
            np.testing.assert_allclose(rad,expected,atol=2e-13)
        for q in curve[::100]:
            for factor,expected in ((.999,'captured'),(1.001,'escaped')):
                xy=center+factor*(q-center)
                assert trace(m,c.event,pixel_momentum(m,c,*xy),settings()).status==expected
    pos=shadow_curve(Kerr(spin=.9),camera(Kerr(spin=.9),90),1001)
    neg=shadow_curve(Kerr(spin=-.9),camera(Kerr(spin=-.9),90),1001)
    assert pos[:,0].min()==pytest.approx(63-neg[:,0].max(),abs=1e-10)
    np.testing.assert_allclose(np.ptp(pos,axis=0),np.ptp(neg,axis=0),atol=1e-10)


def test_horizon_fraction_source_radius_convergence_and_unresolved():
    m=Kerr(spin=.9); c=camera(m)
    p=pixel_momentum(m,c,31.5,23.5)
    a=trace(m,c.event,p,settings(kerr_capture_fraction=.999))
    b=trace(m,c.event,p,settings(kerr_capture_fraction=.98))
    assert a.status==b.status=='captured'
    p=pixel_momentum(m,c,56,30)
    a=trace(m,c.event,p,settings(kerr_source_radius=1e4))
    b=trace(m,c.event,p,settings(kerr_source_radius=1e5))
    assert np.linalg.norm(a.direction-b.direction)<1e-7
    a=trace(m,c.event,p,replace(settings(),max_affine=.1))
    assert a.status=='unresolved' and a.direction is None
    rejected=trace(m,c.event,p,replace(settings(),null_tolerance=1e-16))
    assert rejected.status=='unresolved' and rejected.reason=='null_constraint_exceeded'


def test_config_optimizations_and_spawn(tmp_path):
    from numerical_observatory.cli import load_configuration
    from numerical_observatory.disk import make_disk
    from numerical_observatory.render import render
    config=tmp_path/'kerr.toml'
    config.write_text('mass=2\n[spacetime]\ntype="kerr"\nspin=0.9\n[camera]\nradius=30\ninclination_deg=60\nwidth=4\nheight=3\n[integration]\nescape_radius=70\nmax_affine=350\n',encoding='utf-8')
    m,c,s=load_configuration(config)
    assert m.radius(c.event)==pytest.approx(60)
    assert s.kerr_source_radius==2e5
    scene=make_disk(m,dict(enabled=True))
    assert scene.surface.metric == m and scene.motion.orbit == 'prograde'
    from numerical_observatory.background import CelestialGrid
    from numerical_observatory.symmetry import CenteredRayCache
    from numerical_observatory.schwarzschild_orbit import compatible
    assert not CenteredRayCache(m,c,s).enabled
    assert not compatible(m,c)
    for workers in (1,2):
        report=render(m,c,s,tmp_path/str(workers),background=CelestialGrid(),workers=workers,method='quadrature',samples=2)
        assert report['method']=='ode' and report['rotational_symmetry_reuse'] is False
        assert report['spacetime']['spin']==.9
        assert report['counts']['unresolved']==0
    with np.load(tmp_path/'1/rays.npz') as a,np.load(tmp_path/'2/rays.npz') as b:
        assert {'carter_drift','axial_momentum_drift','kerr_constants'}<=set(a.files)
        for key in a.files:
            np.testing.assert_array_equal(a[key],b[key])


def test_boundary_convergence_and_near_critical_independent_reference():
    from numerical_observatory.kerr_reference import reference_radius,locate_boundary
    m=Kerr(spin=.9); c=camera(m,90); curve=shadow_curve(m,c,20001)
    expected=reference_radius(curve,c,.7)
    measured=[]
    for tol in (1e-7,1e-10):
        measured.append(locate_boundary(m,c,replace(settings(),rtol=tol,atol=tol/100),.7,expected,precision=1e-4))
        assert abs(measured[-1]['error_pixels'])<1e-4
    assert abs(measured[0]['radius_pixels']-measured[1]['radius_pixels'])<1e-4
    assert measured[1]['max_null']<measured[0]['max_null']/5
    # A near-critical exterior ray: separated equations independently resolve
    # the prolonged approach, radial turn and outward escape.
    center=np.array([31.5,23.5]); xy=center+1.0001*expected*np.array([np.cos(.7),np.sin(.7)])
    p=pixel_momentum(m,c,*xy)
    result,sol=trace(m,c.event,p,settings(),return_solution=True)
    assert result.status=='escaped'
    ref=separated_reference(m,np.r_[c.event,p],sol.t[-1])
    assert ref.success
    t=np.linspace(0,sol.t[-1],501)
    radii=np.array([m.radius(y[:4]) for y in sol.sol(t).T])
    np.testing.assert_allclose(radii,ref.sol(t)[0],atol=3e-5,rtol=0)


def test_weak_field_and_captured_radial_coordinate():
    m=Kerr(spin=.99)
    far=np.array([0.,1e5,2e4,3e4])
    assert np.max(abs(m.covariant(far)-ETA))<2.1e-5
    assert np.max(abs(m.covariant(far)-Kerr().covariant(far)))<3e-10
    c=camera(m,20)
    result,sol=trace(m,c.event,pixel_momentum(m,c,31.5,23.5),settings(),return_solution=True)
    assert result.status=='captured'
    assert m.radius(sol.y[:4,-1])==pytest.approx(.995*m.horizon_radius,abs=1e-11)
    assert abs(np.linalg.norm(sol.y[1:4,-1])-.995*m.horizon_radius)>1e-5


def test_schwarzschild_quadrature_limit_and_actual_spin_reflection():
    from numerical_observatory.schwarzschild_orbit import orbit
    zero=Kerr(); c=Camera(position=(0,0,30),observer='zamo',width=3,height=3)
    angle=.25
    f,right,_=c.axes()
    result=trace(zero,c.event,c.momentum(zero,np.cos(angle)*f+np.sin(angle)*right),settings())
    assert result.status=='escaped'
    np.testing.assert_allclose(result.direction,orbit(angle,30).direction,atol=3e-8)
    positive=Kerr(spin=.9); negative=Kerr(spin=-.9)
    cp=camera(positive,60); cn=camera(negative,60)
    a=trace(positive,cp.event,pixel_momentum(positive,cp,56,8),settings())
    b=trace(negative,cn.event,pixel_momentum(negative,cn,7,8),settings())
    assert a.status==b.status=='escaped'
    np.testing.assert_allclose(a.direction,b.direction*np.array([1,-1,1]),atol=2e-8)
