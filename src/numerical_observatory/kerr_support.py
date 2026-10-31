"""Kerr domain policy, observer configuration and far-field continuation.

No spherical-orbit reuse: the general Hamiltonian ODE is used throughout.
"""
from dataclasses import dataclass
import numpy as np
from scipy.integrate import solve_ivp


def inclined_camera_options(metric, options):
    options = dict(options)
    if 'inclination_deg' in options:
        if 'inclination_degrees' in options:
            raise ValueError('specify only one inclination spelling')
        options['inclination_degrees'] = options.pop('inclination_deg')
    if 'radius' in options or 'inclination_degrees' in options:
        if any(k in options for k in ('position','target','up')):
            raise ValueError('Kerr radius/inclination camera cannot also specify position/target/up')
        radius = options.pop('radius',60.)
        angle = options.pop('inclination_degrees',90.)
        if not np.isfinite([radius,angle]).all() or radius <= metric.horizon_radius/metric.mass or not 0 <= angle <= 180:
            raise ValueError('camera radius must exceed horizon and inclination lie in [0,180] degrees')
        theta = np.deg2rad(angle)
        options.update(position=(np.sqrt(radius**2+metric.spin**2)*np.sin(theta),0.,radius*np.cos(theta)),
                       target=(0.,0.,0.),up=(-np.cos(theta),0.,np.sin(theta)))
    options.setdefault('observer','zamo')
    return options


@dataclass(frozen=True)
class KerrDomain:
    metric: object
    settings: object

    def capture(self, event):
        return self.metric.radius(event)-self.settings.kerr_capture_fraction*self.metric.horizon_radius

    def escape(self, event):
        return self.metric.radius(event)-self.settings.escape_radius

    def source_direction(self, state):
        velocity = self.metric.inverse(state[:4]) @ state[4:]
        return velocity[1:]/np.linalg.norm(velocity[1:])


def default_domain(metric,event,settings):
    from .geodesic import TraceDomain
    r = metric.radius(event)
    if r <= metric.horizon_radius:
        raise ValueError('Kerr ray camera must be outside the outer horizon')
    if settings.escape_radius <= max(r,4*metric.mass):
        raise ValueError('escape radius must exceed Kerr camera radius and 4M')
    if settings.kerr_source_radius <= settings.escape_radius:
        raise ValueError('kerr_source_radius must exceed escape_radius')
    policy = KerrDomain(metric,settings)
    return TraceDomain(policy.capture,policy.escape,policy.source_direction)


def sampled_states(sol):
    times = np.sort(np.r_[sol.t,(sol.t[1:]+sol.t[:-1])/2])
    return sol.sol(times).T


def finish_kerr(metric,event,momentum,settings,sol,result,continue_background=True):
    """Audit actual states; escaping sky directions include a far-field ODE tail.

    At R_source use the outgoing Cartesian tangent, not the position vector.
    Remaining bending is O(M*b/R_source**2); radius convergence is audited.
    Tail solves the SAME Hamiltonian equations, with unrestricted step length
    in the weak field. Its constraints, steps and RHS counts are included.
    """
    from .geodesic import hamilton_rhs
    states = sampled_states(sol)
    if result.status == 'escaped' and continue_background:
        def far(_,y):
            return metric.radius(y[:4])-settings.kerr_source_radius
        far.terminal=True; far.direction=1
        # Beyond the escape sphere every selected ray is outward-going. A
        # generous affine interval allows arbitrary initial photon normalization.
        speed = np.linalg.norm((metric.inverse(sol.y[:4,-1]) @ sol.y[4:,-1])[1:])
        span = 4*settings.kerr_source_radius/max(speed,1e-14)
        tail = solve_ivp(lambda t,y:hamilton_rhs(metric,y),(0,span),sol.y[:,-1],
                         method='DOP853',rtol=settings.rtol,atol=settings.atol,
                         events=far,dense_output=True)
        result.steps += len(tail.t)-1; result.evaluations += tail.nfev
        states = np.vstack((states,sampled_states(tail)))
        if not tail.success or not len(tail.t_events[0]):
            result.status='unresolved'; result.reason='kerr_asymptotic_tail_failure'; result.direction=None
        else:
            velocity=metric.inverse(tail.y[:4,-1]) @ tail.y[4:,-1]
            result.direction=velocity[1:]/np.linalg.norm(velocity[1:])
    constants = np.array([metric.invariants(y) for y in states])
    initial=metric.invariants(np.r_[event,momentum])
    scale=max(abs(initial[0]),np.linalg.norm(momentum)*1e-14)
    errors=np.max(np.abs(constants-initial),axis=0)/np.array([scale,metric.mass*scale,(metric.mass*scale)**2])
    result.energy_drift=float(errors[0]); result.angular_momentum_drift=float(errors[1])
    result.carter_drift=float(errors[2]); result.kerr_constants=initial
    null=max(abs(y[4:] @ metric.inverse(y[:4]) @ y[4:])/scale**2 for y in states)
    result.max_null_error=max(result.max_null_error,float(null))
    if not np.isfinite(errors).all() or not np.isfinite(null) or null>settings.null_tolerance:
        result.status='unresolved'; result.reason='null_constraint_exceeded'; result.direction=None
    return result
