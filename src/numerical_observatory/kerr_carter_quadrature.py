"""Validation-only separated Kerr quadrature for inward escaping plane crossings.

Independent of both ODE RHS implementations. Radial roots/divided differences
use standard-library Decimal to investigate near-critical precision; integration
is SciPy adaptive quadrature with a regular turning-point substitution. Not a
third production renderer or a general captured-ray algorithm.
"""
from decimal import Decimal,localcontext
import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from .surfaces import SurfaceHit


def escaping_crossing(metric,event,p,ordinal,precision=50,eps=2e-12):
    E,L,Q=metric.invariants(np.r_[event,p]); energy=abs(E); sign=np.sign(E); M=metric.mass; A=metric.spin
    xi=L/(M*E); eta=Q/(M*E)**2; K=eta+(xi-A)**2; C=A*A-xi*xi-eta
    if eta<=0:raise ValueError('quadrature audit requires equator-crossing eta>0')
    r=metric.radius(event); u0=M/r; mu0=event[3]/r
    s0=np.sqrt(1-mu0*mu0); ptheta=(metric.bl_jacobian(event).T@p)[2]
    w0=-s0*ptheta/(energy*M)
    coefficients=[-A*A*eta,2*K,C,0.,1.]
    roots=np.roots(np.trim_zeros(coefficients,'f'))
    candidates=sorted(z.real for z in roots if abs(z.imag)<1e-8 and z.real>u0)
    if not candidates:raise ValueError('quadrature audit requires an escaping radial turn')
    with localcontext() as context:
        context.prec=precision
        c2=Decimal.from_float(C);c3=Decimal.from_float(2*K);c4=Decimal.from_float(-A*A*eta)
        turning=Decimal.from_float(candidates[0])
        for _ in range(40):
            F=1+c2*turning**2+c3*turning**3+c4*turning**4
            dF=2*c2*turning+3*c3*turning**2+4*c4*turning**3
            next_value=turning-F/dF
            if next_value==turning:break
            turning=next_value
        turn=float(turning)
        def divided(u):
            with localcontext() as ctx:
                ctx.prec=precision
                x=Decimal.from_float(float(u)); t=turning
                return float(-(c2*(x+t)+c3*(x*x+x*t+t*t)+c4*(x**3+x*x*t+x*t*t+t**3)))
        def leg(lower,function=lambda u,v:1.,velocity_sign=1):
            if lower>=turn:return 0.
            width=turn-lower
            def regular(s):
                u=turn-width*s*s
                h=divided(u)
                if h<=0:raise ValueError('nonpositive radial divided difference')
                v=velocity_sign*np.sqrt(max(0.,(turn-u)*h))
                return 2*np.sqrt(width)*function(u,v)/np.sqrt(h)
            return quad(regular,0,1,epsabs=eps,epsrel=eps,limit=300)[0]
        zplus=(2*eta/(np.sqrt(C*C+4*A*A*eta)-C) if C<0 else
               (C+np.sqrt(C*C+4*A*A*eta))/(2*A*A)) if A else -eta/C
        amplitude=np.sqrt(zplus)
        initial=np.arcsin(np.clip(mu0/amplitude,-1,1))
        chi0=initial if w0>=0 else np.pi-initial
        first=int(np.floor(chi0/np.pi+1e-13))+1
        chi_end=(first+ordinal-1)*np.pi
        def polar(function):
            def value(chi):
                mu=amplitude*np.sin(chi)
                return function(mu)/np.sqrt(eta/zplus+A*A*mu*mu)
            points=np.arange(np.floor(chi0/(np.pi/2))+1,np.ceil(chi_end/(np.pi/2)))*np.pi/2
            return quad(value,chi0,chi_end,points=points,epsabs=eps,epsrel=eps,limit=500)[0]
        tau=polar(lambda mu:1.)
        to_turn=leg(u0)
        outward=tau>to_turn
        target=tau-to_turn if outward else to_turn-tau
        lower=1e-12 if outward else u0
        if target>leg(lower):raise ValueError('plane crossing occurs beyond infinity')
        uhit=brentq(lambda u:leg(u)-target,lower,turn,xtol=5e-15,rtol=1e-14)
        def combine(function):
            if outward:return leg(u0,function,1)+leg(uhit,function,-1)
            return leg(u0,function,1)-leg(uhit,function,1)
        def radial_time(u,v):
            B=1+A*(A-xi)*u*u;D=1-2*u+A*A*u*u;Cu=1+A*A*u*u
            return sign*A*(xi-A)+(sign*Cu*B+2*u*v)/(u*u*D)
        def radial_phi(u,v):
            B=1+A*(A-xi)*u*u;D=1-2*u+A*A*u*u;Cu=1+A*A*u*u
            return -sign*A+A*(sign*B+2*u*v/Cu)/D
        T=event[0]+M*(combine(radial_time)+polar(lambda mu:sign*A*A*mu*mu))
        phi=np.arctan2(event[2],event[1])+combine(radial_phi)+polar(lambda mu:sign*xi/(1-mu*mu))
        affine=M*(combine(lambda u,v:1/(energy*u*u))+polar(lambda mu:A*A*mu*mu/energy))
        rhit=M/uhit;rho=np.sqrt(rhit*rhit+metric.a**2)
        hit_event=np.array([T,rho*np.cos(phi),rho*np.sin(phi),0.])
        velocity_sign=-1 if outward else 1
        v=velocity_sign*np.sqrt((turn-uhit)*divided(uhit))
        w=np.cos(chi_end)*np.sqrt(eta)
        D=1-2*uhit+A*A*uhit*uhit
        pbl=np.array([-E,-energy*v/D,-energy*M*w,L])
        momentum=np.linalg.solve(metric.bl_jacobian(hit_event).T,pbl)
        hit=SurfaceHit(affine,hit_event,momentum,ordinal)
        metadata=dict(mino_phase=tau,radial_turn_u=turn,radial_turn_r_M=1/turn,
                      radial_phase_to_turn=to_turn,polar_phase=chi_end-chi0,
                      azimuth_unwrapped=phi,decimal_precision=precision,quad_tolerance=eps)
        return hit,metadata
