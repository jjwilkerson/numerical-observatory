"""Snapshot-guarded SciPy DOP853 stepping of the shared Hamiltonian RHS.

Only numerical metrics use this adapter. Every trial AND dense-output stage
uses the same closed temporal slab. No momentum/cone projection or extrapolation.
"""
from types import SimpleNamespace
import numpy as np
from scipy.integrate import DOP853
from scipy.optimize import brentq
from .numerical_data import NumericalLookupError


class TemporalStage(Exception):
    def __init__(self,affine,time):self.affine=affine;self.time=time


class SlabMetric:
    def __init__(self,parent,slab):
        self.parent=parent;self.slab=slab;self.affine=0.;self.levels=set()

    def evaluate(self,event):
        lo,hi=self.parent.data.times[list(self.slab)]
        tolerance=min(8*np.finfo(float).eps*max(abs(lo),abs(hi)),(hi-lo)/8)
        t=event[0]
        if t<lo-tolerance or t>hi+tolerance:raise TemporalStage(self.affine,t)
        if t<lo or t>hi:event=np.r_[np.clip(t,lo,hi),event[1:]]
        value=self.parent.evaluate(event,self.slab);self.levels.update(self.parent.last_levels)
        return value
    def inverse(self,event):return self.evaluate(event)[0]
    def inverse_derivative(self,event):return self.evaluate(event)[1]


class NumericalDomain:
    def __init__(self,metric):
        self.metric=metric;options=dict(metric.domain_options)
        self.model=options.pop('background','none');self.radius=options.pop('background_radius',None)
        self.center=np.asarray(options.pop('background_center',[0.,0.,0.]),dtype=float)
        self.captures=options.pop('capture_spheres',[])
        self.minimum_step=options.pop('minimum_affine_step',1e-12)
        if not np.isfinite(self.minimum_step) or self.minimum_step<=0:raise ValueError('minimum_affine_step must be positive')
        if options or self.model not in ('none','finite_direction_sphere'):
            raise ValueError('unsupported numerical domain option/background model')
        if self.center.shape!=(3,) or not np.isfinite(self.center).all():raise ValueError('invalid finite background center')
        if self.model=='finite_direction_sphere' and (self.radius is None or not np.isfinite(self.radius) or self.radius<=0):
            raise ValueError('finite direction sphere requires an explicit positive background_radius')
        for b in self.captures:
            if set(b)-{'center','radius','description'} or np.asarray(b.get('center')).shape!=(3,) or not np.isfinite(b['center']).all() or not np.isfinite(b.get('radius',np.nan)) or b['radius']<=0:
                raise ValueError('invalid explicitly supplied numerical capture sphere')

    def capture(self,event):
        return min((np.linalg.norm(event[1:]-np.asarray(b['center']))-b['radius'] for b in self.captures),default=1.)
    def escape(self,event):
        return np.linalg.norm(event[1:]-self.center)-self.radius if self.model!='none' else -1.
    def validate_camera(self,event):
        if self.capture(event)<=0:raise ValueError('camera lies inside a supplied absorber')
        if self.model!='none' and self.escape(event)>=0:raise ValueError('camera must be inside finite background sphere')


class PiecewiseDense:
    def __init__(self,times,states,pieces):self.times=np.asarray(times);self.states=np.asarray(states).T;self.pieces=pieces
    def __call__(self,t):
        values=np.asarray(t);flat=values.reshape(-1)
        if np.any(flat<self.times[0]) or np.any(flat>self.times[-1]):raise ValueError('outside retained numerical trajectory')
        out=np.empty((8,len(flat)))
        for j,value in enumerate(flat):
            exact=np.flatnonzero(self.times==value)
            if len(exact):out[:,j]=self.states[:,exact[0]]
            else:
                index=int(np.searchsorted(self.times,value)-1);out[:,j]=self.pieces[index][0](value)
        return out[:,0] if values.ndim==0 else out.reshape((8,)+values.shape)


def trace_numerical(metric,event,momentum,settings,return_solution=False,surface=None,domain=None):
    from .geodesic import hamilton_rhs,RayResult
    if settings.integrator!='hamiltonian' or surface is not None or domain is not None:
        raise ValueError('numerical metrics require generic Hamiltonian, built-in numerical domain, and no emitting surface')
    policy=NumericalDomain(metric);policy.validate_camera(event)
    state=np.r_[event,momentum];slab=metric.interval(state[0]);direction=np.sign((metric.inverse(event)@momentum)[0])
    if not direction:raise ValueError('zero coordinate-time speed')
    slab=metric.interval(state[0],direction);wrapper=SlabMetric(metric,slab)
    times=[0.];states=[state.copy()];pieces=[];slabs_used=set();rejections=0;lookup_rejections=0;nfev=0
    start_eval=metric.evaluations;start_stencils=metric.stencils;start_failures=dict(metric.failures);reason='affine_limit';status='unresolved';source=None;capture_index=0
    solver=None;complete=True
    def rhs(lam,y):
        nonlocal nfev
        nfev+=1  # Include one-sided endpoint refreshes outside SciPy's fun wrapper.
        wrapper.affine=lam
        return hamilton_rhs(wrapper,y)
    try:
        first_t,last_t=metric.data.times[[0,-1]]
        if (direction<0 and state[0]==first_t) or (direction>0 and state[0]==last_t):
            raise NumericalLookupError('numerical_temporal_boundary','ray starts outward at saved endpoint')
        boundary=metric.data.times[slab[1] if direction>0 else slab[0]]
        speed=(metric.inverse(event)@momentum)[0]
        first=min(settings.max_step,settings.max_affine,abs((boundary-state[0])/speed)/2)
        solver=DOP853(rhs,0.,state,settings.max_affine,rtol=settings.rtol,atol=settings.atol,
                      max_step=settings.max_step,first_step=first)
        for _ in range(1_000_000):
            current=solver.y;slab=metric.interval(current[0],direction);wrapper.slab=slab;slabs_used.add(slab)
            lo,hi=metric.data.times[list(slab)];bound=hi if direction>0 else lo
            tolerance=min(8*np.finfo(float).eps*max(abs(lo),abs(hi)),(hi-lo)/8)
            if (direction<0 and current[0]==first_t) or (direction>0 and current[0]==last_t):
                reason='numerical_temporal_boundary';break
            solver.f=rhs(solver.t,current)
            speed=solver.f[0]
            if speed*direction<=0:raise NumericalLookupError('numerical_time_orientation','coordinate-time direction changed')
            prediction=abs((bound-current[0])/speed)
            solver.h_abs=min(solver.h_abs,prediction)
            old={key:getattr(solver,key) for key in ('t','y','f','h_abs','status','t_old')}
            # A rejected dense-output stage also rolls back the accepted trial.
            for retry in range(1000):
                trial=min(solver.h_abs,settings.max_step,settings.max_affine-old['t'])
                try:
                    message=solver.step()
                    if solver.status=='failed':raise NumericalLookupError('numerical_integration_failure',message)
                    dense=solver.dense_output();break
                except TemporalStage as exc:
                    rejections+=1
                    displacement=exc.time-old['y'][0]
                    fraction=abs((bound-old['y'][0])/displacement) if displacement else 0.
                    guard=1-32*np.finfo(float).eps
                    new_h=trial*np.clip(fraction*guard,.1,guard)
                    for key,value in old.items():setattr(solver,key,value)
                    if not new_h>0 or old['t']+new_h==old['t']:
                        raise NumericalLookupError('numerical_knot_failure','cannot advance to snapshot knot')
                    solver.h_abs=new_h
                except NumericalLookupError:
                    lookup_rejections+=1
                    if trial<=policy.minimum_step*1.01 or old['t']+trial/2==old['t']:raise
                    for key,value in old.items():setattr(solver,key,value)
                    solver.h_abs=max(policy.minimum_step,trial/2)
            else:raise NumericalLookupError('numerical_knot_failure','stage guard retry limit')
            end=solver.y.copy()
            if abs(end[0]-bound)<=tolerance:end[0]=bound
            roots=[]
            for index,fn in enumerate((policy.capture,policy.escape)):
                a,b=fn(old['y'][:4]),fn(end[:4])
                crossed=(a>0 and b<=0) if index==0 else (a<0 and b>=0)
                if crossed:
                    root=brentq(lambda lam:fn(dense(lam)[:4]),old['t'],solver.t,xtol=4*np.finfo(float).eps,rtol=4*np.finfo(float).eps)
                    roots.append((root,index))
            stop=solver.t
            if roots:
                stop,index=min(roots);end=dense(stop)
                if index==0:
                    status='captured';reason='numerical_supplied_capture'
                    capture_index=int(np.argmin([np.linalg.norm(end[1:4]-b['center'])-b['radius'] for b in policy.captures]))+1
                else:status='escaped';reason='numerical_finite_background'
            times.append(stop);states.append(end.copy());pieces.append((dense,slab))
            solver.y=end
            solver.h_abs=max(solver.h_abs,policy.minimum_step)
            if roots or solver.status=='finished':break
        else:reason='numerical_step_limit'
    except NumericalLookupError as exc:
        reason=exc.code;complete=False
    except (ValueError,FloatingPointError,np.linalg.LinAlgError) as exc:
        reason='numerical_integration_failure: '+str(exc);complete=False
    solution=PiecewiseDense(times,states,pieces);scale=max(momentum[0]**2,1e-28);null=0.
    try:
        for i,(dense,slab) in enumerate(pieces):
            for lam in (times[i],(times[i]+times[i+1])/2,times[i+1]):
                z=solution(lam);g=metric.evaluate(z[:4],slab)[0]
                null=max(null,float(abs(z[4:]@g@z[4:])/scale))
        if not pieces:null=float(abs(momentum@metric.inverse(event)@momentum)/scale)
        if not np.isfinite(null) or null>settings.null_tolerance:status='unresolved';reason='numerical_null_constraint';capture_index=0
        if status=='escaped':
            z=states[-1];g=metric.evaluate(z[:4],pieces[-1][1])[0];tangent=(g@z[4:])[1:]
            source=tangent/np.linalg.norm(tangent)
    except NumericalLookupError as exc:
        status='unresolved';reason=exc.code;source=None;capture_index=0;complete=False
    result=RayResult(status,source,null,None,None,len(pieces),nfev,reason,status,
        numerical_diagnostics=dict(capture_index=capture_index,complete=complete,selected_levels=sorted(wrapper.levels),
            evaluations=metric.evaluations-start_eval,stencils=metric.stencils-start_stencils,
            lookup_failures={k:v-start_failures.get(k,0) for k,v in metric.failures.items() if v>start_failures.get(k,0)},
            temporal_stage_rejections=rejections,lookup_stage_rejections=lookup_rejections,
            slabs=len(slabs_used),last_event=states[-1][:4].tolist()))
    sol=SimpleNamespace(t=np.array(times),y=np.array(states).T,sol=solution,nfev=nfev,
                        success=status!='unresolved',message=reason,t_events=[np.array([times[-1]]) if status==label else np.array([]) for label in ('captured','escaped')])
    return (result,sol) if return_solution else result
