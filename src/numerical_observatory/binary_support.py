"""Finite-sphere sky mapping and provisional binary capture/validity policy."""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class BinaryDomain:
    metric: object
    settings: object

    def capture_distances(self,event):
        return np.array([np.linalg.norm(event[1:]-np.asarray(c))-2*m*self.settings.binary_capture_fraction
                         if m>0 else np.inf for m,c in zip(self.metric.masses,self.metric.centers)])

    def capture(self,event):
        return float(self.capture_distances(event).min())

    def escape(self,event):
        return np.linalg.norm(event[1:]-self.metric.barycenter)-self.settings.escape_radius

    def valid_region(self,event):
        return self.metric.signature_margin(event)-self.settings.binary_signature_margin

    def source_direction(self,state):
        velocity=self.metric.inverse(state[:4])@state[4:]
        return velocity[1:]/np.linalg.norm(velocity[1:])


def default_domain(metric,event,settings):
    from .geodesic import TraceDomain
    policy=BinaryDomain(metric,settings)
    if policy.capture(event)<=0 or policy.valid_region(event)<=0:
        raise ValueError('binary camera must be outside provisional capture regions and signature guard')
    enclosing=max(np.linalg.norm(np.asarray(c)-metric.barycenter)+2*m for m,c in zip(metric.masses,metric.centers) if m>0)
    if settings.escape_radius<=max(enclosing,np.linalg.norm(event[1:]-metric.barycenter)):
        raise ValueError('binary escape sphere must enclose camera and both provisional capture spheres')
    return TraceDomain(policy.capture,policy.escape,policy.source_direction,policy.valid_region)


def finish_binary(metric,settings,sol,result):
    policy=BinaryDomain(metric,settings)
    times=np.sort(np.r_[sol.t,(sol.t[1:]+sol.t[:-1])/2])
    states=sol.sol(times).T
    margins=np.array([metric.signature_margin(z[:4]) for z in states])
    scale=max(abs(states[0,4]),np.linalg.norm(states[0,4:])*1e-14)
    result.energy_drift=float(np.max(abs(states[:,4]-states[0,4]))/scale)
    capture_index=0
    if result.status=='captured':
        capture_index=int(np.argmin(policy.capture_distances(sol.y[:4,-1])))+1
        result.reason=f'provisional_capture_{capture_index}'
    if result.reason=='data_domain_exit':result.reason='binary_signature_guard'
    if not np.isfinite(margins).all() or margins.min()<settings.binary_signature_margin-1e-8:
        result.status='unresolved';result.reason='binary_invalid_signature_sample';result.direction=None
        capture_index=0
    result.binary_diagnostics=dict(capture_index=capture_index,provisional_capture=capture_index>0,
                                  minimum_signature_margin=float(margins.min()),complete=True)
    return result
