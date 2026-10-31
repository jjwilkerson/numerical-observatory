"""Independent Kerr references: separated Carter equations and finite-camera shadow.

Spherical photon constants follow Teo (2003), obtained from R=R'=0.
Local critical covectors are transformed from BL to the ACTUAL camera tetrad;
no distant-observer alpha/beta coordinates are compared directly with pixels.
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq


def potentials(metric, r, theta, E, L, Q):
    a=metric.a; M=metric.mass
    delta=r*r-2*M*r+a*a
    P=E*(r*r+a*a)-a*L
    K=Q+(L-a*E)**2
    R=P*P-delta*K
    dR=4*E*r*P-2*(r-M)*K
    s=np.sin(theta); c=np.cos(theta)
    theta_potential=Q+a*a*E*E*c*c-L*L*c*c/(s*s)
    derivative=-2*a*a*E*E*c*s+2*L*L*c/(s**3)
    return R,dR,theta_potential,derivative


def separated_reference(metric, state, duration, rtol=1e-12, atol=1e-13):
    """Validation-only second-order separated equations, parameterized by affine.

    Mino radial/polar velocities evolve through zero without square-root sign
    switching, resolving both kinds of turning point. BL singularities restrict
    this independent audit to the exterior, away from the coordinate axis.
    """
    r=metric.radius(state[:4]); theta=np.arccos(state[3]/r)
    pbl=metric.bl_jacobian(state[:4]).T @ state[4:]
    E,L,Q=metric.invariants(state)
    delta=r*r-2*metric.mass*r+metric.a**2
    initial=[r,delta*pbl[1],theta,pbl[2],np.arctan2(state[2],state[1]),state[0]]
    def rhs(_,y):
        r,v,theta,w,phi,t=y
        a=metric.a; M=metric.mass
        sigma=r*r+a*a*np.cos(theta)**2
        delta=r*r-2*M*r+a*a
        P=E*(r*r+a*a)-a*L
        _,dR,_,dTheta=potentials(metric,r,theta,E,L,Q)
        dphi=L/np.sin(theta)**2-a*E+a*P/delta-2*M*a*r/((r*r+a*a)*delta)*v
        dt=a*(L-a*E*np.sin(theta)**2)+(r*r+a*a)*P/delta-2*M*r/delta*v
        return np.array([v,dR/2,w,dTheta/2,dphi,dt])/sigma
    return solve_ivp(rhs,(0,duration),initial,method='DOP853',rtol=rtol,atol=atol,dense_output=True,max_step=2*metric.mass)


def spherical_constants(metric,r):
    M,a=metric.mass,metric.a
    xi=(r*r*(r-3*M)+a*a*(r+M))/(a*(M-r))
    eta=r**3*(4*a*a*M-r*(r-3*M)**2)/(a*a*(M-r)**2)
    return xi,eta


def sky_to_pixel(camera, direction):
    f,right,up=camera.axes()
    scale=camera.width/(2*np.tan(np.deg2rad(camera.fov_degrees)/2))
    depth=direction@f
    return np.array([(camera.width-1)/2+scale*(direction@right)/depth,
                     (camera.height-1)/2-scale*(direction@up)/depth])


def pixel_momentum(metric,camera,x,y):
    f,right,up=camera.axes()
    half=np.tan(np.deg2rad(camera.fov_degrees)/2)
    return camera.momentum(metric,f+(2*(x+.5)/camera.width-1)*half*right
                           +(1-2*(y+.5)/camera.height)*half*camera.height/camera.width*up)


def shadow_curve(metric,camera,count=1001):
    """Critical boundary in zero-based pixel-center coordinates, finite observer.

    Cartesian KS r is oblate; p_BL is pulled back through the full coordinate
    Jacobian, including time and azimuth radial shifts. Both polar signs used.
    """
    r0=metric.radius(camera.event)
    if r0 <= 5*metric.mass:
        raise ValueError('analytical shadow comparison requires camera outside photon region (r>5M)')
    if abs(metric.spin)<1e-7:
        from .metric import Schwarzschild
        if metric.spin != 0:
            raise ValueError('use nondegenerate spin or exactly zero for analytical curve')
        radial=np.asarray(camera.position)/r0
        if camera.observer_frame is not None or np.linalg.norm(camera.axes()[0]+radial)>1e-12:
            raise ValueError('zero-spin analytical reference requires a centered static/ZAMO camera')
        angle=Schwarzschild(metric.mass).shadow_angle(r0)
        f,right,up=camera.axes()
        return np.array([sky_to_pixel(camera,np.cos(angle)*f+np.sin(angle)*(np.cos(t)*right+np.sin(t)*up))
                         for t in np.linspace(0,2*np.pi,count,endpoint=False)])
    theta=np.arccos(camera.position[2]/r0)
    if abs(np.sin(theta))<1e-6:
        raise ValueError('finite shadow reference presently excludes exact polar-axis observers; renderer supports them')
    spin=abs(metric.spin); M=metric.mass
    low=2*M*(1+np.cos(2/3*np.arccos(-spin)))
    high=2*M*(1+np.cos(2/3*np.arccos(spin)))
    def theta_p(r):
        xi,eta=spherical_constants(metric,r)
        return eta+metric.a**2*np.cos(theta)**2-xi*xi/np.tan(theta)**2
    middle=brentq(lambda r:spherical_constants(metric,r)[0],low,high,xtol=1e-13*M)
    lower=low if theta_p(low)>=0 else brentq(theta_p,low,middle,xtol=1e-13*M)
    upper=high if theta_p(high)>=0 else brentq(theta_p,middle,high,xtol=1e-13*M)
    # Cosine spacing resolves the endpoints where polar momentum has a square root.
    radial=(lower+upper)/2-(upper-lower)/2*np.cos(np.linspace(0,np.pi,(count+1)//2))
    frame=camera.tetrad(metric); jac=metric.bl_jacobian(camera.event)
    delta=r0*r0-2*M*r0+metric.a**2
    points=[]
    for sign,rs in ((1,radial),(-1,radial[::-1])):
        for r in rs:
            xi,eta=spherical_constants(metric,r)
            E=-1.; L=E*xi; Q=eta
            R,_,Theta,_=potentials(metric,r0,theta,E,L,Q)
            if R<=0:
                raise ValueError('critical ray cannot reach camera')
            pbl=np.array([-E,-np.sqrt(R)/delta,sign*np.sqrt(max(0.,Theta)),L])
            p=np.linalg.solve(jac.T,pbl)
            local=frame[1:]@p/(frame[0]@p)
            points.append(sky_to_pixel(camera,local))
    return np.array(points)


def reference_radius(curve,camera,angle):
    center=np.array([(camera.width-1)/2,(camera.height-1)/2])
    offsets=curve-center
    phi=np.arctan2(offsets[:,1],offsets[:,0])
    order=np.argsort(phi)
    return float(np.interp(angle,phi[order],np.linalg.norm(offsets,axis=1)[order],period=2*np.pi))


def locate_boundary(metric,camera,settings,angle,reference,precision=1e-4):
    """Bracket capture/escape then bisect, independently of critical constants.

    The reference radius only seeds a bracket. Every bisection decision uses
    an actual production Hamiltonian trajectory, not a radial-potential oracle.
    Far-field background continuation is unnecessary for capture classification.
    Returns the midpoint AND final bracket so sampling uncertainty is explicit.
    """
    from .geodesic import trace
    from .kerr_support import default_domain
    center=np.array([(camera.width-1)/2,(camera.height-1)/2])
    direction=np.array([np.cos(angle),np.sin(angle)])
    domain=default_domain(metric,camera.event,settings)
    evaluations=0; max_null=0.; max_carter=0.
    def classification(radius):
        nonlocal evaluations,max_null,max_carter
        p=pixel_momentum(metric,camera,*(center+radius*direction))
        result=trace(metric,camera.event,p,settings,domain=domain)
        evaluations+=1
        max_null=max(max_null,result.max_null_error)
        max_carter=max(max_carter,result.carter_drift)
        if result.status not in ('captured','escaped'):
            raise RuntimeError(f'unresolved boundary ray: {result.reason}; '
                               f'null={result.max_null_error:.6g}, radius={radius:.12g} pixels')
        return result.status
    width=max(.2,reference*.04)
    for _ in range(12):
        lo=max(.01,reference-width); hi=reference+width
        if classification(lo)=='captured' and classification(hi)=='escaped':
            break
        width*=2
    else:
        raise RuntimeError('cannot bracket Kerr shadow boundary')
    while hi-lo>precision:
        mid=(hi+lo)/2
        if classification(mid)=='captured': lo=mid
        else: hi=mid
    return dict(radius_pixels=(lo+hi)/2,bracket_pixels=[lo,hi],
                error_pixels=(lo+hi)/2-reference,ray_count=evaluations,
                max_null=max_null,max_carter=max_carter)
