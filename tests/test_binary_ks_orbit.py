from dataclasses import replace
import numpy as np
import pytest
from numerical_observatory.trajectories import CircularOrbit,InertialPair
from numerical_observatory.boosted_ks import lorentz_boost
from numerical_observatory.binary_ks_orbit_approx import BinaryKSOrbitApprox
from numerical_observatory.binary_ks_approx import BinaryKSApprox
from numerical_observatory.metric import ETA,Schwarzschild
from numerical_observatory.camera import Camera
from numerical_observatory.geodesic import Integration,TraceDomain,trace,hamilton_rhs
from numerical_observatory.render import render
from numerical_observatory.cli import load_configuration
from numerical_observatory.background import CelestialGrid


def settings(**kw):
    return Integration(**(dict(rtol=1e-10,atol=1e-12,escape_radius=250,max_affine=800,max_step=20)|kw))


def test_newtonian_unequal_worldlines_and_regime_rejections():
    orbit=CircularOrbit((.7,.3),30,23,5,(1,2,3))
    for t in (-30,5,70):
        x,v,a=orbit.state(t)
        np.testing.assert_allclose(np.asarray(orbit.masses)@x,orbit.barycenter,atol=1e-14)
        assert np.linalg.norm(x[1]-x[0])==pytest.approx(30)
        assert np.linalg.norm(v[0])==pytest.approx(9*np.sqrt(1/30**3))
        h=1e-3
        np.testing.assert_allclose(v,(orbit.state(t+h)[0]-orbit.state(t-h)[0])/(2*h),atol=2e-12)
        np.testing.assert_allclose(a,(orbit.state(t+h)[1]-orbit.state(t-h)[1])/(2*h),atol=2e-13)
    for kwargs in (dict(masses=(0,1)),dict(separation=12),dict(separation=-1),dict(rate_scale=1000)):
        with pytest.raises(ValueError):CircularOrbit(**kwargs)
    with pytest.raises(ValueError):InertialPair(velocities=((1.,0,0),(0,0,0)))
    with pytest.raises(ValueError):BinaryKSOrbitApprox((.7,.3),orbit.__class__())


@pytest.mark.parametrize('velocity',[(0,0,0),(.15,-.08,.1),(-.2,0,0)])
def test_exact_single_boost_tensors_and_all_derivatives(velocity):
    center=(2,-1,.5);epoch=3
    m=BinaryKSOrbitApprox((1,0),InertialPair((center,(0,0,0)),(velocity,(0,0,0)),epoch))
    L=lorentz_boost(velocity);inv=np.linalg.inv(L);ref=Schwarzschild()
    np.testing.assert_allclose(L.T@ETA@L,ETA,atol=3e-16)
    for event in ([7,5,4,2],[-20,10,-3,8],[25,1,4,30]):
        x=np.asarray(event,dtype=float);X=L@(x-np.r_[epoch,center])
        expected=L.T@ref.covariant(X)@L
        np.testing.assert_allclose(m.covariant(x),expected,atol=8e-16)
        np.testing.assert_allclose(m.inverse(x),inv@ref.inverse(X)@inv.T,atol=9e-16)
        expected_d=np.einsum('ba,mi,nj,bij->amn',L,inv,inv,ref.inverse_derivative(X))
        np.testing.assert_allclose(m.inverse_derivative(x),expected_d,atol=4e-16)


def test_circular_analytic_derivatives_signature_and_initial_nullness():
    m=BinaryKSOrbitApprox()
    for event in ([0,15,1.5,0],[27,-10,4,2],[-20,2,3,80]):
        x=np.asarray(event,dtype=float);g=m.covariant(x)
        assert sum(np.linalg.eigvalsh(g)<0)==1
        np.testing.assert_allclose(g@m.inverse(x),np.eye(4),atol=2e-15)
        for a in range(4):
            dx=np.eye(4)[a]
            cs=np.imag(m.inverse(x+1j*1e-20*dx))/1e-20
            np.testing.assert_allclose(m.inverse_derivative(x)[a],cs,atol=2e-14,rtol=0)
            errors=[]
            for h in (1e-2,1e-3,1e-4):
                fd=(-m.inverse(x+2*h*dx)+8*m.inverse(x+h*dx)-8*m.inverse(x-h*dx)+m.inverse(x-2*h*dx))/(12*h)
                errors.append(np.max(abs(fd-cs)))
            assert min(errors)<2e-10
        # Independent generalized determinant lemma using boosted null vectors.
        from numerical_observatory.boosted_ks import rest_position
        s,v,_=m.body_states(x[0]);terms=[]
        for mass,position,vel in zip(m.masses,s,v):
            X=rest_position(x,position,vel);r=np.linalg.norm(X)
            k=lorentz_boost(vel).T@np.r_[1,-X/r]
            assert abs(k@ETA@k)<1e-14
            terms.append((2*mass/r,k))
        D=1-terms[0][0]*terms[1][0]*(terms[0][1]@ETA@terms[1][1])**2
        assert m.signature_margin(x)==pytest.approx(D,abs=2e-15)
    c=Camera(position=(0,0,75),time=17,width=4,height=3)
    f=c.tetrad(m);np.testing.assert_allclose(f@m.covariant(c.event)@f.T,ETA,atol=1e-14)
    for _,_,p in c.rays(m):assert abs(p@m.inverse(c.event)@p)<2e-14 and p[0]>0


def test_consistent_multiple_time_zero_rate_limit():
    base=CircularOrbit(phase_degrees=19);stationary=BinaryKSApprox(base.masses,tuple(map(tuple,base.state(0)[0])))
    errors=[]
    for rate in (.1,.01,.001,0):
        m=BinaryKSOrbitApprox(trajectory=replace(base,rate_scale=rate));error=0.
        for t in (-70,0,100):
            x=np.array([t,9,2,5.])
            error=max(error,np.max(abs(m.covariant(x)-stationary.covariant(x))))
            if rate==0:
                np.testing.assert_allclose(m.inverse_derivative(x),stationary.inverse_derivative(x),atol=2e-16)
        errors.append(error)
    assert errors[1]<errors[0]/5 and errors[2]<errors[1]/5 and errors[3]<2e-16
    c=Camera(position=(0,0,75));s=settings();p=c.momentum(stationary,[.15,.05,-1])
    h=trace(stationary,c.event,p,s)
    a=trace(BinaryKSOrbitApprox(trajectory=replace(base,rate_scale=0)),c.event,p,s)
    assert a.status==h.status=='escaped'
    np.testing.assert_allclose(a.direction,h.direction,atol=2e-10)


def test_transformed_exact_capture_and_escape_trajectories():
    v=(.15,-.07,0);L=lorentz_boost(v);inv=np.linalg.inv(L);ref=Schwarzschild()
    moving=BinaryKSOrbitApprox((1,0),InertialPair(velocities=(v,(0,0,0))))
    rc=Camera(position=(0,0,20));X=rc.event;x=inv@X;s=settings(escape_radius=150,max_affine=500)
    frame=rc.tetrad(ref)@inv.T
    c=Camera(position=tuple(x[1:]),time=x[0],observer_frame=tuple(map(tuple,frame)))
    np.testing.assert_allclose(c.tetrad(moving)@moving.covariant(x)@c.tetrad(moving).T,ETA,atol=2e-15)
    def sky(z):
        tangent=inv@(ref.inverse(z[:4])@z[4:]);return tangent[1:]/np.linalg.norm(tangent[1:])
    domain=TraceDomain(lambda e:np.linalg.norm(e[1:])-1.99,
                       lambda e:np.linalg.norm((inv@e)[1:])-s.escape_radius,sky)
    for direction in ([0,0,-1],[.4,0,-1],[0,.7,-1]):
        p=rc.momentum(ref,direction);global_p=L.T@p
        np.testing.assert_allclose(c.momentum(moving,direction),global_p,atol=2e-15)
        a,sa=trace(moving,x,global_p,s,return_solution=True)
        b,sb=trace(ref,X,p,s,domain=domain,return_solution=True)
        assert a.status==b.status and a.status!='unresolved'
        for t in np.linspace(0,min(sa.t[-1],sb.t[-1]),31):
            z=sb.sol(t);expected=np.r_[inv@z[:4],L.T@z[4:]]
            np.testing.assert_allclose(sa.sol(t),expected,atol=3e-7,rtol=0)
        if a.direction is not None:np.testing.assert_allclose(a.direction,b.direction,atol=1e-8,rtol=0)


def test_helical_identity_pt_evolution_and_equal_mass_phase_symmetries():
    m=BinaryKSOrbitApprox();w=m.trajectory.helical_omega;x=np.array([17.,13,2,4])
    K=np.array([1.,-w*x[2],w*x[1],0]);A=np.zeros((4,4));A[1,2]=-w;A[2,1]=w
    g=m.covariant(x)
    np.testing.assert_allclose(np.einsum('a,aij->ij',K,m.covariant_derivative(x))+A.T@g+g@A,0,atol=5e-17)
    half=x.copy();half[0]+=np.pi/w
    np.testing.assert_allclose(m.covariant(half),g,atol=3e-15)
    theta=.3;R=np.eye(4);R[1:3,1:3]=[[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]]
    y=R@x;y[0]+=theta/w
    np.testing.assert_allclose(m.covariant(y),R@g@R.T,atol=2e-15)
    c=Camera(position=(0,0,75));p=c.momentum(m,[.15,.05,-1]);b,sol=trace(m,c.event,p,settings(),return_solution=True)
    assert b.status=='escaped' and b.energy_drift is None and b.binary_diagnostics['energy_change']>1e-3
    assert b.binary_diagnostics['helical_error']<1e-8 and np.all(np.diff(sol.y[0])<0)
    t=sol.t[-1]/3;z=sol.sol(t);h=1e-3
    rate=(sol.sol(t+h)[4]-sol.sol(t-h)[4])/(2*h)
    direct=-.5*z[4:]@m.inverse_derivative(z[:4])[0]@z[4:]
    assert hamilton_rhs(m,z)[4]==pytest.approx(direct,abs=1e-16)
    assert rate==pytest.approx(direct,abs=1e-8)


def test_config_invalid_backends_and_unresolved(tmp_path):
    path=tmp_path/'cfg.toml';path.write_text('mass=2\n[spacetime]\ntype="binary_ks_orbit_approx"\nmasses=[0.7,0.3]\nseparation=30\nepoch=5\nphase_degrees=27\n[camera]\ntime=17\n')
    m,c,s=load_configuration(path)
    assert m.masses==(1.4,.6) and m.trajectory.separation==60 and m.trajectory.epoch==10 and c.time==34
    with pytest.raises(ValueError):render(m,c,replace(s,integrator='carter'),tmp_path/'invalid')
    with pytest.raises(ValueError):render(m,c,s,tmp_path/'invalid',method='quadrature')
    with pytest.raises(ValueError):Camera(position=(0,0,75),observer='zamo').tetrad(m)
    from numerical_observatory.disk import make_disk
    with pytest.raises(ValueError):make_disk(m,{'enabled':True})
    c=replace(c,position=(0,0,75));p=c.momentum(m,[.1,0,-1])
    b=trace(m,c.event,p,settings(max_affine=.01))
    assert b.status=='unresolved' and not b.binary_diagnostics['provisional_capture']


def test_spawn_supersampling_determinism_and_saved_time_diagnostics(tmp_path):
    m=BinaryKSOrbitApprox();c=Camera(position=(0,0,75),time=31,width=4,height=3,fov_degrees=40)
    for workers in (1,2):
        report=render(m,c,settings(),tmp_path/str(workers),background=CelestialGrid(),workers=workers,samples=2)
        assert report['spacetime']['type']=='binary_ks_orbit_approx' and not report['rotational_symmetry_reuse']
        assert not report['stationary_energy_conserved'] and report['counts']['unresolved']==0
        assert report['maxima']['helical_error']<1e-7
    with np.load(tmp_path/'1/rays.npz') as a,np.load(tmp_path/'2/rays.npz') as b:
        for key in a.files:np.testing.assert_array_equal(a[key],b[key])
        assert np.isnan(a['energy_drift']).all() and np.isfinite(a['energy_change']).all()


def test_overlap_signature_and_curvature_controls():
    # Diagnostic controls deliberately violate the circular model's wide-orbit
    # restriction to exercise invalid-domain handling, not physical binaries.
    c=Camera(position=(0,0,20));s=settings(max_step=2)
    overlap=BinaryKSOrbitApprox(trajectory=InertialPair(centers=((0,0,0),(0,0,0))))
    b=trace(overlap,c.event,c.momentum(overlap,[0,0,-1]),s)
    assert b.status=='unresolved' and b.reason=='moving_capture_overlap'
    invalid=BinaryKSOrbitApprox(trajectory=InertialPair(centers=((-1,0,0),(1,0,0))))
    b=trace(invalid,c.event,c.momentum(invalid,[0,0,-1]),replace(s,max_step=.02))
    assert b.status=='unresolved' and b.reason=='orbit_binary_signature_guard'
    from numerical_observatory.curvature_diagnostic import ricci
    exact=BinaryKSOrbitApprox((1,0),InertialPair(velocities=((.15,.03,0),(0,0,0))))
    residuals=[ricci(exact,[0,3,1,2],h)['max_coordinate_component'] for h in (.03,.01,.003)]
    assert residuals[-1]<1e-11 and residuals[1]<residuals[0]/30
    approximate=BinaryKSOrbitApprox()
    values=[ricci(approximate,[0,15,2,1],h)['scalar'] for h in (.01,.003)]
    assert abs(values[0]-values[1])<1e-8 and abs(values[-1])>1e-4


def test_orbit_ray_radius_convergence_and_half_period_capture_labels():
    m=BinaryKSOrbitApprox();c=Camera(position=(0,0,75));s=settings()
    p=c.momentum(m,[.26,-.09,-1]);rays=[]
    for tol in (1e-7,1e-9,1e-11):
        rays.append(trace(m,c.event,p,replace(s,rtol=tol,atol=tol/100)))
    assert all(r.status=='escaped' for r in rays)
    assert np.linalg.norm(rays[1].direction-rays[2].direction)<1e-7
    sky=[trace(m,c.event,p,replace(s,rtol=1e-11,atol=1e-13,escape_radius=r,max_affine=3000,max_step=40)) for r in (250,500,1000)]
    assert all(r.status=='escaped' for r in sky)
    assert np.linalg.norm(sky[1].direction-sky[2].direction)<.8*np.linalg.norm(sky[0].direction-sky[1].direction)
    direction=[.18,-.09,-1];p=c.momentum(m,direction)
    capture=trace(m,c.event,p,s)
    later=replace(c,time=np.pi/m.trajectory.helical_omega)
    other=trace(m,later.event,later.momentum(m,direction),s)
    assert capture.binary_diagnostics['capture_index']==1 and other.binary_diagnostics['capture_index']==2
    for fraction in (.98,.999):
        same=trace(m,c.event,p,replace(s,binary_capture_fraction=fraction))
        assert same.status==capture.status=='captured' and same.binary_diagnostics['capture_index']==1


def test_nonconstant_speed_acceleration_derivatives():
    # A local analytic control exercises gamma_dot, which vanishes for circular
    # motion. It is not a supported inspiral prescription or a render scene.
    class AcceleratingControl:
        epoch=0.;barycenter=(0,0,0);helical_omega=None
        def state(self,t):
            return (np.array([[.1*t+.005*t*t,.03*t,0],[30,0,0]]),
                    np.array([[.1+.01*t,.03,0],[0,0,0]]),np.array([[.01,0,0],[0,0,0]]))
    m=BinaryKSOrbitApprox((1,0),AcceleratingControl());x=np.array([2.,4,3,2])
    dx=np.array([1.,0,0,0])
    independent=np.imag(m.inverse(x+1j*1e-20*dx))/1e-20
    np.testing.assert_allclose(m.inverse_derivative(x)[0],independent,atol=1e-15)
