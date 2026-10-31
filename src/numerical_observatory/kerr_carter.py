"""Exact-Kerr Carter backend: compact radial/polar second-order equations.

u=M/r, mu=cos(theta), tau=|E| M gamma. The evolved velocities are du/dtau
and dmu/dtau. Affine/M is the solver independent variable, preserving max_step.
No square-root branch switching occurs in the encounter. Horizon-regular KS
rates use algebraically equivalent on-shell expressions, whose constraints
and metric momentum/velocity consistency are independently audited.
"""
from dataclasses import replace
from types import SimpleNamespace
import numpy as np
from scipy.integrate import solve_ivp
from .kerr import Kerr
from .surfaces import CoplanarRay, KerrEquatorialAnnulus


class CarterFailure(RuntimeError):
    pass


def compatible_domain(metric, settings, domain):
    if domain is None: return True
    from .kerr_support import KerrDomain
    policy=getattr(domain.capture,'__self__',None)
    return (isinstance(policy,KerrDomain) and policy.metric==metric
            and policy.settings.kerr_capture_fraction==settings.kerr_capture_fraction
            and policy.settings.escape_radius==settings.escape_radius
            and domain.escape==policy.escape and domain.source_direction==policy.source_direction
            and domain.valid_region is None)


def validate_backend(metric,settings,domain=None,surface=None):
    if settings.integrator!='carter': return
    if type(metric) is not Kerr:
        raise ValueError('Carter integrator requires the exact Kerr metric; Schwarzschild quadrature remains separate')
    if not compatible_domain(metric,settings,domain):
        raise ValueError('Carter integrator requires the standard Kerr domain; custom domains use Hamiltonian')
    if surface is not None and (type(surface) is not KerrEquatorialAnnulus or surface.metric!=metric):
        raise ValueError('Carter surface tracing requires a matching equatorial Kerr annulus')
    if surface is not None and settings.escape_radius<=surface.outer_radius:
        raise ValueError('Carter disk tracing requires escape_radius beyond the emitting outer edge')


class CarterOrbit:
    def __init__(self,metric,event,momentum):
        self.metric=metric; self.M=metric.mass; self.A=metric.spin
        self.main_evaluations=0;self.tail_evaluations=0
        self.event=np.asarray(event,dtype=float).copy(); self.p=np.asarray(momentum,dtype=float).copy()
        self.constants=metric.invariants(np.r_[event,momentum])
        E,L,Q=self.constants; self.eps=abs(E); self.sign=np.sign(E)
        if self.eps<1e-12*np.linalg.norm(momentum):
            raise CarterFailure('near_zero_Killing_energy')
        self.xi=L/(self.M*E); self.eta=Q/(self.M*E)**2
        self.K=self.eta+(self.xi-self.A)**2
        self.C=self.A*self.A-self.xi*self.xi-self.eta
        if self.A:
            disc=self.C*self.C+4*self.A*self.A*self.eta
            if disc < -128*np.finfo(float).eps*(1+self.C*self.C):
                raise CarterFailure('invalid_polar_potential')
            root=np.sqrt(max(0.,disc))
            zplus=(2*self.eta/(root-self.C) if self.C<0 and self.eta>0
                   else (self.C+root)/(2*self.A*self.A))
        else:
            zplus=-self.eta/self.C if self.C else 0.
        self.polar_scale=max(1e-8,min(1.,1-zplus))
        r=metric.radius(event); mu=event[3]/r; u=self.M/r
        if 1-mu*mu<1e-12:
            raise CarterFailure('polar_axis_initial_condition')
        velocity=metric.inverse(event)@momentum
        xyz=event[1:]; disc=r*r+metric.a**2*event[3]**2/r**2
        dr=r*xyz/disc; dr=dr.copy(); dr[2]+=metric.a**2*event[3]/(r*disc)
        radial=float(dr@velocity[1:])
        sigma_factor=1+self.A*self.A*mu*mu*u*u
        v=-radial*sigma_factor/self.eps
        w=(velocity[3]-mu*radial)*sigma_factor/(self.eps*u)
        self.initial=np.array([u,v,mu,w,np.arctan2(event[2],event[1]),event[0]/self.M,0.])

    def F(self,u):
        return 1+self.C*u*u+2*self.K*u**3-self.A*self.A*self.eta*u**4

    def G(self,mu):
        return self.eta+self.C*mu*mu-self.A*self.A*mu**4

    def angular_rates(self,u,v,mu):
        A=self.A; sign=self.sign; s2=1-mu*mu
        if np.any(s2<1e-14): raise CarterFailure('polar_axis_coordinate_limit')
        B=1+A*(A-self.xi)*u*u; C=1+A*A*u*u; D=1-2*u+A*A*u*u
        plus=v+sign*B; minus=v-sign*B
        # Avoid BL 0/0 at the horizon. Use (v+sign B)/D = -K u²/(v-sign B)
        # on the null first integral, ONLY on the cancellation-safe branch.
        stable=np.abs(plus)<np.abs(minus)
        if np.ndim(u)==0:
            if stable:
                if abs(minus)<1e-15: raise CarterFailure('degenerate_radial_coordinate')
                h=-self.K*u*u/minus
            else:
                if abs(D)<1e-15: raise CarterFailure('outgoing_horizon_chart_limit')
                h=plus/D
        else:
            h=np.empty_like(np.asarray(u))
            h[stable]=-self.K*u[stable]**2/minus[stable]
            h[~stable]=plus[~stable]/D[~stable]
        phi=sign*(self.xi/s2-A)+sign*A*B/C+2*A*u*h/C
        time=sign*(A*(self.xi-A*s2)+B/(u*u))+2*h/u
        radial_covector=self.eps*(-h+sign*B/C)
        return phi,time,radial_covector

    def mino_rhs(self,y):
        u,v,mu,w,phi,T,tau=y
        if u<=0 or not np.isfinite(y).all(): raise CarterFailure('invalid_separated_state')
        phi_rate,time_rate,_=self.angular_rates(u,v,mu)
        return np.array([v,self.C*u+3*self.K*u*u-2*self.A*self.A*self.eta*u**3,
                         w,self.C*mu-2*self.A*self.A*mu**3,phi_rate,time_rate,1.])

    def rhs(self,_,y):
        self.main_evaluations+=1
        u,_,mu,_,_,_,_=y
        factor=self.eps*u*u/(1+self.A*self.A*mu*mu*u*u)
        return factor*self.mino_rhs(y)

    def cartesian(self,y):
        y=np.asarray(y); u,v,mu,w,phi,T,tau=y
        if np.any(u<=0) or np.any(1-mu*mu<1e-14): raise CarterFailure('invalid_reconstruction_coordinates')
        r=self.M/u; s=np.sqrt(1-mu*mu); den=r*r+(self.A*self.M)**2
        rho=np.sqrt(den)*s; co=np.cos(phi); si=np.sin(phi)
        _,_,q=self.angular_rates(u,v,mu)
        pt=-self.constants[0]+np.zeros_like(u)
        ptheta=-self.eps*self.M*w/s
        sigma=r*r+(self.A*self.M)**2*mu*mu
        prho=np.sqrt(den)*(r*s*q+mu*ptheta)/sigma
        pz=(den*mu*q-r*s*ptheta)/sigma
        azimuthal=self.constants[1]/rho
        return np.array([T*self.M,rho*co,rho*si,r*mu,pt,prho*co-azimuthal*si,
                         prho*si+azimuthal*co,pz])

    def tangent(self,y):
        u,v,mu,w,phi,T,tau=y
        rates=self.rhs(0,y); du,_,dmu,_,dphi,dT,_=rates
        r=self.M/u; dr=-self.M*du/u**2
        rho=np.sqrt(r*r+self.metric.a**2)*np.sqrt(1-mu*mu)
        drho=r*dr/(r*r+self.metric.a**2)*rho-mu*dmu/(1-mu*mu)*rho
        # derivative wrt physical affine lambda (rhs uses lambda/M).
        return np.array([dT,(drho*np.cos(phi)-rho*np.sin(phi)*dphi)/self.M,
                         (drho*np.sin(phi)+rho*np.cos(phi)*dphi)/self.M,
                         (dr*mu+r*dmu)/self.M])

    def dense_adapter(self,sol,stop=None):
        times=sol.t*self.M
        if stop is not None: times=np.r_[times[times<stop],stop]
        def dense(t):
            t=np.asarray(t)
            if stop is not None and np.any(t>stop+1e-10):
                raise ValueError('opaque ray terminates at the first emitting surface')
            value=self.cartesian(sol.sol(t/self.M))
            if t.ndim==0:
                if t==0: return np.r_[self.event,self.p].copy()
            else:
                value[:,t==0]=np.r_[self.event,self.p][:,None]
            return value
        events=[x*self.M for x in sol.t_events]
        if stop is not None:
            events=[x[x<=stop] for x in events]+[np.array([stop])]
        return SimpleNamespace(t=times,sol=dense,y=dense(times),
            t_events=events,success=sol.success,message=sol.message,
            nfev=sol.nfev,separated_solution=sol,carter_orbit=self)

    def audit(self,ys):
        states=self.cartesian(ys).T
        actual=np.array([self.metric.invariants(z) for z in states])
        scales=np.array([self.eps,self.M*self.eps,(self.M*self.eps)**2])
        drift=np.max(abs(actual-self.constants),axis=0)/scales
        null=max(abs(z[4:]@self.metric.inverse(z[:4])@z[4:])/self.eps**2 for z in states)
        radial=float(np.max(abs(ys[1]**2-self.F(ys[0]))))
        polar=float(np.max(abs(ys[3]**2-self.G(ys[2]))))
        return float(null),drift,radial,polar


def angular_tail(orbit,y,settings):
    """Exact separated angular equations on monotone outward inverse radius.

    The known outward leg has v=-sqrt(F); there is no turning point in this
    tail. Stop at the SAME finite R_source as Hamiltonian. No straight-line
    bending approximation is introduced. KS time is immaterial for stationary
    sky mapping and is not represented as an evolved tail event.
    """
    def rhs(u,z):
        orbit.tail_evaluations+=1
        mu,w,phi=z
        F=orbit.F(u)
        if F<=0: raise CarterFailure('nonmonotone_background_tail')
        v=-np.sqrt(F)
        phi_rate,_,_=orbit.angular_rates(u,v,mu)
        return [w/v,(orbit.C*mu-2*orbit.A**2*mu**3)/v,phi_rate/v]
    final_u=orbit.M/settings.kerr_source_radius
    polar_rtol=max(100*np.finfo(float).eps,settings.rtol*orbit.polar_scale)
    sol=solve_ivp(rhs,(y[0],final_u),y[[2,3,4]],method='DOP853',rtol=[polar_rtol,polar_rtol,settings.rtol],
                  atol=np.array([settings.atol/orbit.M*y[0]*orbit.polar_scale,
                                 settings.atol/orbit.M*orbit.polar_scale,settings.atol/orbit.M*y[0]]),dense_output=True)
    if not sol.success: raise CarterFailure('background_tail_solver_failure')
    us=np.sort(np.r_[sol.t,(sol.t[:-1]+sol.t[1:])/2])
    polar=sol.sol(us)
    ys=np.array([us,-np.sqrt(orbit.F(us)),polar[0],polar[1],polar[2],
                 np.full_like(us,y[5]),np.full_like(us,y[6])])
    # Samples are sorted in ascending inverse radius; the source endpoint is
    # therefore FIRST, since the outward integration runs toward decreasing u.
    final=orbit.cartesian(ys[:,0]); velocity=orbit.metric.inverse(final[:4])@final[4:]
    direction=velocity[1:]/np.linalg.norm(velocity[1:])
    return direction,ys,sol


def trace_carter(metric,event,momentum,settings,domain=None,return_solution=False,surface=None):
    from .geodesic import RayResult,trace
    from .kerr_support import default_domain
    validate_backend(metric,settings,domain,surface)
    default_domain(metric,event,settings)  # same physical-domain validation
    orbit=None; sol=None; attempted_evaluations=0; attempted_steps=0
    null=float(abs(momentum @ metric.inverse(event) @ momentum)/max(momentum[0]**2,1e-28))
    drift=np.zeros(3);radial=polar=None
    try:
        orbit=CarterOrbit(metric,event,momentum)
        ucap=metric.mass/(settings.kerr_capture_fraction*metric.horizon_radius)
        uesc=metric.mass/settings.escape_radius
        def capture(t,y): return ucap-y[0]
        def escape(t,y): return uesc-y[0]
        capture.terminal=True;capture.direction=-1
        escape.terminal=True;escape.direction=1
        tolerance=settings.atol/metric.mass
        min_u=min(orbit.initial[0],uesc)
        atols=tolerance*np.array([min_u**2,1.,min_u,1.,min_u,1.,1.])
        atols[2:4]*=orbit.polar_scale
        rtols=np.full(7,settings.rtol)
        rtols[2:4]=max(100*np.finfo(float).eps,settings.rtol*orbit.polar_scale)
        sol=solve_ivp(orbit.rhs,(0,settings.max_affine/metric.mass),orbit.initial,
            method='DOP853',rtol=rtols,atol=atols,max_step=settings.max_step/metric.mass,
            events=[capture,escape],dense_output=True)
        attempted_evaluations=sol.nfev;attempted_steps=len(sol.t)-1
        if not sol.success: raise CarterFailure('encounter_solver_failure: '+sol.message)
        adapter=orbit.dense_adapter(sol)
        hit=None; coplanar=False
        if surface is not None:
            try: hit=surface.first_hit(adapter)
            except CoplanarRay: coplanar=True
        stop=sol.t[-1] if hit is None else hit.affine/metric.mass
        accepted=np.r_[sol.t[sol.t<stop],stop]
        samples=np.sort(np.r_[accepted,(accepted[:-1]+accepted[1:])/2])
        ys=sol.sol(samples)
        null,drift,radial,polar=orbit.audit(ys)
        status,reason,direction='unresolved','affine_limit',None
        if coplanar: reason='coplanar_surface'
        elif hit is not None: status,reason='emitted','first_opaque_surface'
        elif len(sol.t_events[0]): status,reason='captured','capture_boundary'
        elif len(sol.t_events[1]): status,reason='escaped','outward_escape'
        tail_diag=None
        if status=='escaped' and domain is None:
            direction,tail_ys,tail=angular_tail(orbit,ys[:,-1],settings)
            attempted_evaluations+=tail.nfev;attempted_steps+=len(tail.t)-1
            tn,td,tr,tp=orbit.audit(tail_ys)
            null=max(null,tn);drift=np.maximum(drift,td);radial=max(radial,tr);polar=max(polar,tp)
            tail_diag=dict(steps=len(tail.t)-1,evaluations=tail.nfev,source_radius=settings.kerr_source_radius)
        elif status=='escaped':
            final=adapter.y[:,-1]; velocity=metric.inverse(final[:4])@final[4:]
            direction=velocity[1:]/np.linalg.norm(velocity[1:])
        # Actual metric constraints plus independent first integrals. Do not
        # certify conserved constants merely because they are RHS parameters.
        if not np.isfinite([null,radial,polar,*drift]).all(): raise CarterFailure('nonfinite_constraints')
        # Guard the separatrix with measured radial first-integral uncertainty.
        extrema=np.roots([-2*orbit.A**2*orbit.eta,3*orbit.K,orbit.C]) if orbit.A**2*orbit.eta else np.roots([3*orbit.K,orbit.C])
        margins=[]
        for root in extrema:
            relevant=(status in ('captured','escaped') or root.real<=float(np.max(ys[0]))
                      or (orbit.initial[1]>0 and float(np.min(ys[1]))<0))
            if abs(root.imag)<1e-10 and orbit.initial[1]>0 and orbit.initial[0]<root.real<=ucap and relevant:
                margin=abs(orbit.F(root.real));margins.append(float(margin))
                scale=1+abs(orbit.C*root.real**2)+abs(2*orbit.K*root.real**3)+abs(orbit.A**2*orbit.eta*root.real**4)
                dq_sensitivity=abs(-root.real**2+2*root.real**3-orbit.A**2*root.real**4)
                if margin<=radial+dq_sensitivity*drift[2]+64*np.finfo(float).eps*scale:
                    raise CarterFailure('near_critical_radial_uncertainty')
        if null>settings.null_tolerance: raise CarterFailure('null_constraint_exceeded')
        if radial>settings.null_tolerance or drift[2]>settings.null_tolerance:
            raise CarterFailure('separated_constraint_exceeded')
        diag=dict(radial_first_integral_error=radial,polar_first_integral_error=polar,
                  separatrix_margins=margins,tail=tail_diag,parameter='affine/M with normalized Mino velocities',
                  max_mino_phase=float(ys[6].max()),polar_tolerance_scale=orbit.polar_scale)
        result=RayResult(status,direction,float(null),float(drift[0]),float(drift[1]),attempted_steps,
                         attempted_evaluations,reason,status,hit=hit if status=='emitted' else None,
                         carter_drift=float(drift[2]),kerr_constants=orbit.constants,
                         backend='carter',carter_diagnostics=diag)
        answer=orbit.dense_adapter(sol,None if result.hit is None else result.hit.affine)
        return (result,answer) if return_solution else result
    except CarterFailure as exc:
        if orbit is not None:
            attempted_evaluations=max(attempted_evaluations,orbit.main_evaluations+orbit.tail_evaluations)
        if sol is not None:
            attempted_evaluations=max(attempted_evaluations,sol.nfev)
            attempted_steps=max(attempted_steps,len(sol.t)-1)
        failure=str(exc)
        # A finite Hamiltonian run cannot certify a mathematically critical
        # orbit either. Keep this explicit uncertainty unresolved, even when
        # ordinary coordinate/numerical failures request Hamiltonian fallback.
        if settings.carter_fallback=='hamiltonian' and failure!='near_critical_radial_uncertainty':
            answer=trace(metric,event,momentum,replace(settings,integrator='hamiltonian'),domain=domain,
                         return_solution=return_solution,surface=surface)
            result=answer[0] if return_solution else answer
            result.backend='hamiltonian';result.fallback_reason=failure
            result.evaluations+=attempted_evaluations;result.steps+=attempted_steps
            result.carter_diagnostics=dict(failed=True,attempted_evaluations=attempted_evaluations)
            return answer
        constants=metric.invariants(np.r_[event,momentum])
        result=RayResult('unresolved',None,float(null),float(drift[0]),float(drift[1]),attempted_steps,attempted_evaluations,
                         'carter_failure: '+failure,carter_drift=float(drift[2]),kerr_constants=constants,
                         backend='carter',carter_diagnostics=dict(failed=True,reason=failure,
                             constraint_diagnostics_complete=radial is not None,
                             radial_first_integral_error=radial,polar_first_integral_error=polar))
        answer=None if orbit is None or sol is None else orbit.dense_adapter(sol)
        return (result,answer) if return_solution else result
