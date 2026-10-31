"""Independent Kerr disk checks in BL coordinates and the timelike radial potential.

References: Bardeen, Press & Teukolsky (1972), ApJ 178, 347; Carter separated
radial equations. This module is validation-only, not a production renderer.
"""
import numpy as np
from scipy.optimize import brentq
from .emitter_motion import kerr_orbit_sign


def circular_bl(metric,r,orbit='prograde'):
    """Solve radial geodesic equilibrium using BL metric radial derivatives.

    This does not call the production closed-form Omega or metric derivatives.
    Returns Omega, ut, specific E and coordinate Lz.
    """
    M,a=metric.mass,metric.a
    s=kerr_orbit_sign(metric,orbit)
    gtt=-1+2*M/r; gtp=-2*M*a/r; gpp=r*r+a*a+2*M*a*a/r
    dtt=-2*M/r**2; dtp=2*M*a/r**2; dpp=2*r-2*M*a*a/r**2
    omega=(-dtp+s*np.sqrt(dtp**2-dtt*dpp))/dpp
    ut=1/np.sqrt(-(gtt+2*omega*gtp+omega**2*gpp))
    return np.array([omega,ut,-ut*(gtt+omega*gtp),ut*(gtp+omega*gpp)])


def circular_integrals(metric,r,orbit='prograde'):
    """Independent closed-form specific E,Lz for comparison with g.u."""
    s=kerr_orbit_sign(metric,orbit); v=np.sqrt(metric.mass/r); spin=metric.spin
    den=np.sqrt(1-3*v*v+2*s*spin*v**3)
    E=(1-2*v*v+s*spin*v**3)/den
    L=metric.mass*s*(1-2*s*spin*v**3+spin*spin*v**4)/(v*den)
    return E,L


def massive_radial(metric,r,E,L):
    a,M=metric.a,metric.mass
    delta=r*r-2*M*r+a*a; ddelta=2*(r-M)
    P=E*(r*r+a*a)-a*L; K=(L-a*E)**2
    R=P*P-delta*(r*r+K)
    dR=4*E*r*P-ddelta*(r*r+K)-2*r*delta
    ddR=4*E*P+8*E*E*r*r-2*(r*r+K)-4*r*ddelta-2*delta
    return np.array([R,dR,ddR])


def independent_isco(metric,orbit='prograde'):
    """Locate marginal stability R''=0, without the Z1/Z2 ISCO formula."""
    s=kerr_orbit_sign(metric,orbit)
    photon=2*metric.mass*(1+np.cos(2/3*np.arccos(-s*metric.spin)))
    return brentq(lambda r: massive_radial(metric,r,*circular_integrals(metric,r,orbit))[2],
                  photon*(1+1e-4),20*metric.mass,xtol=1e-12*metric.mass)


def circular_residual(metric,motion,event):
    """Canonical geodesic RHS vs exact derivative along the circular worldline."""
    from .geodesic import hamilton_rhs
    u=motion.four_velocity(event); p=metric.covariant(event)@u
    speed=motion.omega(metric.radius(event))*u[0]
    expected_p=np.array([0.,-speed*p[2],speed*p[1],0.])
    actual=hamilton_rhs(metric,np.r_[event,p])
    return float(np.max(abs(actual-np.r_[u,expected_p])))


def bl_frequency_shift(metric,camera,initial_p,hit_radius,orbit='prograde'):
    """Independent positive past-photon frequencies for a ZAMO or static camera.

    No emitter KS momentum contraction: use conserved E,Lz and BL circular
    orbit equilibrium, plus the camera's actual finite-radius convention.
    """
    E,L,_=metric.invariants(np.r_[camera.event,initial_p])
    r=metric.radius(camera.event); c=camera.position[2]/r
    sigma=r*r+metric.a**2*c*c; delta=r*r-2*metric.mass*r+metric.a**2
    if camera.observer=='zamo' and camera.observer_frame is None:
        A=(r*r+metric.a**2)**2-metric.a**2*delta*(1-c*c)
        omega_o=2*metric.mass*metric.a*r/A; ut_o=np.sqrt(A/(sigma*delta))
    elif camera.observer=='static' and camera.observer_frame is None:
        omega_o=0.; ut_o=1/np.sqrt(1-2*metric.mass*r/sigma)
    else:
        raise ValueError('independent BL audit requires a configured ZAMO or static observer')
    omega_e,ut_e,_,_=circular_bl(metric,hit_radius,orbit)
    observer_frequency=ut_o*(-E+omega_o*L)
    emitter_frequency=ut_e*(-E+omega_e*L)
    if observer_frequency<=0 or emitter_frequency<=0:
        raise ValueError('nonpositive independent photon frequency')
    return float(observer_frequency/emitter_frequency)


def separated_hits(metric,camera,initial_p,duration,rtol=1e-12,atol=1e-13):
    """Independently locate equatorial crossings on a Carter-equation trajectory."""
    from .kerr_reference import separated_reference
    from .surfaces import SurfaceHit
    ref=separated_reference(metric,np.r_[camera.event,initial_p],duration,rtol=rtol,atol=atol)
    if not ref.success:
        raise RuntimeError('separated exterior Kerr reference failed: '+ref.message)
    times=np.sort(np.r_[ref.t,(ref.t[1:]+ref.t[:-1])/2]); levels=np.cos(ref.sol(times)[2])
    E,L,_=metric.invariants(np.r_[camera.event,initial_p]); roots=[]
    for lo,hi,flo,fhi in zip(times[:-1],times[1:],levels[:-1],levels[1:]):
        if flo*fhi<0:
            root=brentq(lambda t:np.cos(ref.sol(t)[2]),lo,hi,xtol=1e-11)
            if root>1e-10 and (not roots or root-roots[-1].affine>1e-9):
                r,v,theta,w,phi,T=ref.sol(root)
                xyz=np.array([np.sqrt(r*r+metric.a**2)*np.sin(theta)*np.cos(phi),
                              np.sqrt(r*r+metric.a**2)*np.sin(theta)*np.sin(phi),r*np.cos(theta)])
                event=np.r_[T,xyz]; delta=r*r-2*metric.mass*r+metric.a**2
                momentum=np.linalg.solve(metric.bl_jacobian(event).T,np.array([-E,v/delta,w,L]))
                roots.append(SurfaceHit(root,event,momentum,len(roots)+1))
    return roots
