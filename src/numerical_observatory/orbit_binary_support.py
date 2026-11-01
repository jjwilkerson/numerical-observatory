"""Moving provisional absorbers and audits for instantaneous boosted KS metrics."""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class OrbitBinaryDomain:
    metric: object
    settings: object

    def capture_distances(self,event):
        radii=self.metric.rest_radii(event)
        return radii-2*np.asarray(self.metric.masses)*self.settings.binary_capture_fraction

    def capture(self,event):return float(self.capture_distances(event).min())
    def escape(self,event):return np.linalg.norm(event[1:]-self.metric.barycenter)-self.settings.escape_radius
    def enclosure_margin(self,event):
        positions,_,_=self.metric.body_states(event[0])
        extent=max(np.linalg.norm(s-self.metric.barycenter)+2*m for s,m in zip(positions,self.metric.masses) if m)
        return 1-extent/self.settings.escape_radius

    def valid_region(self,event):
        return min(self.metric.signature_margin(event)-self.settings.binary_signature_margin,self.enclosure_margin(event))

    def source_direction(self,state):
        velocity=self.metric.inverse(state[:4])@state[4:]
        return velocity[1:]/np.linalg.norm(velocity[1:])


def default_domain(metric,event,settings):
    from .geodesic import TraceDomain
    policy=OrbitBinaryDomain(metric,settings)
    if policy.capture(event)<=0 or policy.valid_region(event)<=0:
        raise ValueError('orbiting binary camera must be outside provisional absorbers and valid-domain guards')
    if policy.escape(event)>=0:raise ValueError('escape sphere must enclose the camera')
    return TraceDomain(policy.capture,policy.escape,policy.source_direction,policy.valid_region)


def finish_orbit_binary(metric,settings,sol,result):
    policy=OrbitBinaryDomain(metric,settings)
    times=np.sort(np.r_[sol.t,(sol.t[1:]+sol.t[:-1])/2]);states=sol.sol(times).T
    margins=np.array([metric.signature_margin(z[:4]) for z in states])
    enclosures=np.array([policy.enclosure_margin(z[:4]) for z in states])
    scale=max(abs(states[0,4]),np.linalg.norm(states[0,4:])*1e-14)
    change=float(np.max(abs(states[:,4]-states[0,4]))/scale)
    helical=metric.helical_quantity(states)
    error=None if helical is None else float(np.max(abs(helical-helical[0]))/scale)
    result.energy_drift=None  # p_t is NOT a conserved quantity
    capture_index=0
    if result.status=='captured':
        distances=policy.capture_distances(sol.y[:4,-1])
        if np.count_nonzero(distances<=1e-8*metric.mass)>1:
            result.status='unresolved';result.reason='moving_capture_overlap';result.direction=None
        else:
            capture_index=int(np.argmin(distances))+1
            result.reason=f'provisional_moving_capture_{capture_index}'
    if result.reason=='data_domain_exit':
        result.reason='orbit_binary_enclosure_guard' if enclosures[-1]<1e-8 else 'orbit_binary_signature_guard'
    if not np.isfinite(margins).all() or margins.min()<settings.binary_signature_margin-1e-8:
        result.status='unresolved';result.reason='orbit_binary_invalid_signature_sample';result.direction=None;capture_index=0
    if error is not None and (not np.isfinite(error) or error>settings.null_tolerance):
        result.status='unresolved';result.reason='helical_constraint_exceeded';result.direction=None;capture_index=0
    result.binary_diagnostics=dict(capture_index=capture_index,provisional_capture=capture_index>0,
                                  minimum_signature_margin=float(margins.min()),complete=True,
                                  energy_change=change,helical_error=error,stationary_energy_conserved=False)
    return result
