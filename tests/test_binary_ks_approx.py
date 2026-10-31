from dataclasses import replace
import numpy as np
import pytest
from numerical_observatory.binary_ks_approx import BinaryKSApprox
from numerical_observatory.binary_support import BinaryDomain
from numerical_observatory.metric import Schwarzschild,ETA
from numerical_observatory.camera import Camera
from numerical_observatory.cli import load_configuration
from numerical_observatory.geodesic import Integration,trace
from numerical_observatory.render import render
from numerical_observatory.background import CelestialGrid


def settings(**kw):
    return Integration(**(dict(rtol=1e-10,atol=1e-12,escape_radius=200,max_affine=600,max_step=10)|kw))


def test_covariant_superposition_inverse_derivatives_signature():
    m=BinaryKSApprox((.7,.3),((-6,1,0),(6,-1,0)))
    for x in ([0,0,0,0],[7,-6,2.5,0],[0,8,3,4],[100,0,0,100]):
        x=np.array(x,dtype=float)
        expected=ETA.copy()
        for mass,center in zip(m.masses,m.centers):
            local=x.copy();local[1:]-=center
            expected+=Schwarzschild(mass).covariant(local)-ETA
        np.testing.assert_allclose(m.covariant(x),expected,atol=1e-15)
        np.testing.assert_allclose(m.covariant(x)@m.inverse(x),np.eye(4),atol=2e-15)
        assert np.count_nonzero(np.linalg.eigvalsh(expected)<0)==1
        assert -np.linalg.det(expected)==pytest.approx(m.signature_margin(x),abs=1e-15)
        for a in range(4):
            dx=np.eye(4)[a]*1e-5
            np.testing.assert_allclose(m.inverse_derivative(x)[a],(m.inverse(x+dx)-m.inverse(x-dx))/2e-5,atol=2e-9,rtol=0)
            np.testing.assert_allclose(m.covariant_derivative(x)[a],(m.covariant(x+dx)-m.covariant(x-dx))/2e-5,atol=2e-9,rtol=0)
    # Explicitly demonstrate why the isolated inverse corrections cannot be summed.
    x=np.zeros(4);wrong=ETA.copy()
    for mass,center in zip(m.masses,m.centers):
        wrong+=Schwarzschild(mass).inverse(np.r_[0,x[1:]-center])-ETA
    assert np.max(abs(wrong-m.inverse(x)))>1e-3


@pytest.mark.parametrize('center',[(0,0,0),(2,-1,3)])
def test_single_hole_limit_metric_tetrad_and_trajectories(center):
    m=BinaryKSApprox((1,0),(center,(0,0,0)));ref=Schwarzschild()
    c=Camera(position=tuple(np.asarray(center)+[0,0,20]),target=center,width=3,height=3)
    rc=replace(c,position=(0,0,20),target=(0,0,0))
    np.testing.assert_allclose(c.tetrad(m),rc.tetrad(ref),atol=1e-14)
    s=settings(capture_radius=1.99)
    for direction in ([0,0,-1],[.45,0,-1],[0,.7,-1]):
        p=c.momentum(m,direction)
        h,hs=trace(ref,rc.event,rc.momentum(ref,direction),s,return_solution=True)
        b,bs=trace(m,c.event,p,s,return_solution=True)
        assert b.status==h.status and b.status!='unresolved'
        for t in np.linspace(0,min(bs.t[-1],hs.t[-1]),17):
            state=bs.sol(t);state[1:4]-=center
            np.testing.assert_allclose(state,hs.sol(t),atol=2e-7,rtol=0)
        if b.status=='escaped':
            v=ref.inverse(hs.y[:4,-1])@hs.y[4:,-1]
            np.testing.assert_allclose(b.direction,v[1:]/np.linalg.norm(v[1:]),atol=2e-8,rtol=0)
        x=c.event+np.array([1,2,1,3]);local=x.copy();local[1:]-=center
        np.testing.assert_allclose(m.covariant(x),ref.covariant(local),atol=1e-15)
        np.testing.assert_allclose(m.inverse(x),ref.inverse(local),atol=1e-15)
        np.testing.assert_allclose(m.inverse_derivative(x),ref.inverse_derivative(local),atol=1e-15)


def test_zero_mass_center_is_regular_and_exchange_symmetry():
    m=BinaryKSApprox((0,1),((0,0,0),(10,0,0)))
    assert np.isfinite(m.inverse(np.zeros(4))).all()
    m=BinaryKSApprox();ex=BinaryKSApprox(m.masses[::-1],m.centers[::-1])
    reflection=np.diag([1,-1,1,1])
    for x in ([0,2,3,4],[0,-5,2,0]):
        x=np.asarray(x,dtype=float)
        np.testing.assert_allclose(m.covariant(x),ex.covariant(x),atol=1e-15)
        np.testing.assert_allclose(m.covariant(reflection@x),reflection@m.covariant(x)@reflection,atol=1e-15)


def test_camera_tetrad_nullness_and_stationary_energy():
    m=BinaryKSApprox();c=Camera(position=(3,5,40),width=5,height=3)
    frame=c.tetrad(m)
    np.testing.assert_allclose(frame@m.covariant(c.event)@frame.T,ETA,atol=2e-14)
    for _,_,p in c.rays(m):
        assert p[0]>0
        assert abs(p@m.inverse(c.event)@p)<2e-14
    b=trace(m,c.event,c.momentum(m,[.3,.1,-1]),settings())
    assert b.status!='unresolved' and b.energy_drift<1e-12


def test_provisional_captures_both_centers_and_affine_limit():
    m=BinaryKSApprox();c=Camera(position=(0,0,45),fov_degrees=40,width=5,height=3)
    result=[trace(m,c.event,p,settings()) for r,_,p in c.rays(m) if r==1]
    assert [b.binary_diagnostics['capture_index'] for b in result]==[0,1,0,2,0]
    assert result[1].reason=='provisional_capture_1' and result[3].reason=='provisional_capture_2'
    np.testing.assert_allclose(result[0].direction*np.array([-1,1,1]),result[4].direction,atol=1e-11,rtol=0)
    b=trace(m,c.event,c.momentum(m,[0,0,-1]),settings(max_affine=.01))
    assert b.status=='unresolved' and b.binary_diagnostics['capture_index']==0 and b.direction is None


def test_nonlorentzian_region_is_not_capture():
    m=BinaryKSApprox(centers=((-1,0,0),(1,0,0)))
    assert m.signature_margin(np.zeros(4))<0
    assert np.all(np.linalg.eigvalsh(m.covariant(np.zeros(4)))>0)
    c=Camera(position=(0,0,10))
    b=trace(m,c.event,c.momentum(m,[0,0,-1]),settings(max_step=.02))
    assert b.status=='unresolved' and b.reason=='binary_signature_guard'
    assert not b.binary_diagnostics['provisional_capture'] and b.direction is None


def test_integration_failure_is_explicit(monkeypatch):
    m=BinaryKSApprox();c=Camera(position=(0,0,30));p=c.momentum(m,[.3,0,-1])
    original=BinaryKSApprox.inverse_derivative
    def failure(self,x):
        if x[3]<29.99:raise np.linalg.LinAlgError('deliberate inversion failure')
        return original(self,x)
    monkeypatch.setattr(BinaryKSApprox,'inverse_derivative',failure)
    b=trace(m,c.event,p,settings())
    assert b.status=='unresolved' and 'deliberate inversion failure' in b.reason
    assert not b.binary_diagnostics['complete'] and not b.binary_diagnostics['provisional_capture']


def test_tolerance_and_escape_radius_convergence():
    m=BinaryKSApprox();c=Camera(position=(0,0,45))
    p=c.momentum(m,[.14,.07,-1])
    rays=[trace(m,c.event,p,settings(rtol=t,atol=t/100)) for t in (1e-7,1e-9,1e-11)]
    assert all(r.status=='escaped' for r in rays)
    assert np.linalg.norm(rays[1].direction-rays[2].direction)<np.linalg.norm(rays[0].direction-rays[2].direction)
    directions=[trace(m,c.event,p,settings(escape_radius=r,max_affine=2500,max_step=20)).direction for r in (200,400,800)]
    assert np.linalg.norm(directions[1]-directions[2])<np.linalg.norm(directions[0]-directions[1])/2
    # Single-hole source agrees with the existing infinity-map as the sphere grows.
    one=BinaryKSApprox((1,0),((0,0,0),(6,0,0)));c=replace(c,position=(0,0,20))
    p=c.momentum(one,[.4,0,-1]);ref=trace(Schwarzschild(),c.event,p,settings()).direction
    errors=[np.linalg.norm(trace(one,c.event,p,settings(escape_radius=r,max_affine=2500,max_step=20)).direction-ref) for r in (200,400,800)]
    assert errors[2]<errors[1]<errors[0] and errors[2]<1e-4


def test_configuration_and_rejections(tmp_path):
    cfg=tmp_path/'binary.toml'
    cfg.write_text('mass=2.0\n[spacetime]\ntype="binary_ks_approx"\nmasses=[0.7,0.3]\ncenters=[[-6,0,0],[6,0,0]]\n')
    m,c,s=load_configuration(cfg)
    assert m.masses==(1.4,.6) and m.centers[0]==(-12,0,0) and c.position==(0,0,40)
    assert s.escape_radius==200
    from numerical_observatory.disk import make_disk
    with pytest.raises(ValueError):make_disk(m,{'enabled':True})
    with pytest.raises(ValueError):render(m,c,replace(s,integrator='carter'),tmp_path/'invalid')
    with pytest.raises(ValueError):render(m,c,s,tmp_path/'invalid',method='quadrature')
    with pytest.raises(ValueError):trace(m,c.event,c.momentum(m,[0,0,-1]),s,domain=object())
    for masses in ((0,0),(-1,2),(float('nan'),1)):
        with pytest.raises(ValueError):BinaryKSApprox(masses)
    with pytest.raises(ValueError):Camera(position=(6,0,1)).tetrad(BinaryKSApprox())


def test_serial_spawn_supersampling_and_saved_provisional_data(tmp_path):
    m=BinaryKSApprox();c=Camera(position=(0,0,45),fov_degrees=40,width=6,height=4)
    reports=[]
    for workers in (1,2):
        report=render(m,c,settings(),tmp_path/str(workers),background=CelestialGrid(),workers=workers,samples=2)
        reports.append(report)
        assert report['spacetime']['type']=='binary_ks_approx' and not report['rotational_symmetry_reuse']
        assert report['integrated_rays']==96 and report['counts']['unresolved']==0
        assert sum(report['provisional_capture_counts'].values())==report['counts']['captured']
    with np.load(tmp_path/'1/rays.npz') as a,np.load(tmp_path/'2/rays.npz') as b:
        for key in a.files:np.testing.assert_array_equal(a[key],b[key])
        assert a['status'].shape==(8,12)
        assert np.all(a['capture_index'][a['status']==0]>0)
        assert np.all(a['capture_index'][a['status']!=0]==0)
        assert np.all(a['constraint_diagnostics_complete'])
