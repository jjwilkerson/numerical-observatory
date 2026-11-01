"""Prescribed global-coordinate worldlines; independent of spacetime geometry."""
from dataclasses import dataclass
from typing import Protocol
import numpy as np


class Trajectory(Protocol):
    barycenter: tuple
    epoch: float
    helical_omega: float | None
    def state(self,time): ...  # positions, velocities, accelerations, each (2,3)
    def description(self): ...


@dataclass(frozen=True)
class CircularOrbit:
    masses: tuple = (.5,.5)
    separation: float = 30.
    phase_degrees: float = 0.
    epoch: float = 0.
    barycenter: tuple = (0.,0.,0.)
    rate_scale: float = 1.  # explicit zero-rate/convergence CONTROL, not a PN term

    def __post_init__(self):
        m=np.asarray(self.masses,dtype=float)
        if m.shape!=(2,) or not np.isfinite(m).all() or np.any(m<=0):
            raise ValueError('circular orbit requires two positive finite masses')
        if (not np.isfinite([self.separation,self.phase_degrees,self.epoch,self.rate_scale]).all()
                or self.separation<=0 or self.rate_scale<0):
            raise ValueError('invalid circular separation, phase, epoch or control rate')
        if self.separation<20*m.sum():
            raise ValueError('leading-order circular model is restricted to separation >=20 M_total')
        if np.asarray(self.barycenter).shape!=(3,) or not np.isfinite(self.barycenter).all():
            raise ValueError('barycenter must be a finite three-vector')
        if max(abs(self.radii))*self.helical_omega>=1:
            raise ValueError('superluminal circular velocity')

    @property
    def radii(self):
        m1,m2=self.masses
        return np.array([self.separation*m2/(m1+m2),-self.separation*m1/(m1+m2)])

    @property
    def helical_omega(self):
        return self.rate_scale*np.sqrt(sum(self.masses)/self.separation**3)

    def state(self,time):
        phase=np.deg2rad(self.phase_degrees)+self.helical_omega*(time-self.epoch)
        unit=np.array([np.cos(phase),np.sin(phase),0.])
        tangent=np.array([-np.sin(phase),np.cos(phase),0.])
        offset=self.radii[:,None]*unit
        return (offset+np.asarray(self.barycenter),self.helical_omega*self.radii[:,None]*tangent,
                -self.helical_omega**2*offset)

    def description(self):
        positions,velocities,accelerations=self.state(self.epoch)
        return dict(model='leading_order_circular' if self.rate_scale==1 else 'rate_scaled_circular_control',
                    separation=self.separation,phase_degrees=self.phase_degrees,epoch=self.epoch,
                    barycenter=list(self.barycenter),omega=float(self.helical_omega),rate_scale=self.rate_scale,
                    period=float(2*np.pi/self.helical_omega) if self.helical_omega else None,
                    initial_positions=positions.tolist(),initial_velocities=velocities.tolist(),
                    initial_accelerations=accelerations.tolist(),orbital_speeds=np.linalg.norm(velocities,axis=1).tolist(),
                    angular_momentum_axis='+Z',regime_policy='d >=20 M_total; not an accuracy guarantee',
                    inspiral=False)


@dataclass(frozen=True)
class InertialPair:
    """Exact single-moving-hole and stationary controls, not orbital dynamics."""
    centers: tuple = ((0.,0.,0.),(30.,0.,0.))
    velocities: tuple = ((0.,0.,0.),(0.,0.,0.))
    epoch: float = 0.
    barycenter: tuple = (0.,0.,0.)  # fixed reference origin for finite escape sphere
    helical_omega = None

    def __post_init__(self):
        if (np.asarray(self.centers).shape!=(2,3) or np.asarray(self.velocities).shape!=(2,3)
                or not np.isfinite(self.centers).all() or not np.isfinite(self.velocities).all()
                or not np.isfinite(self.epoch) or np.asarray(self.barycenter).shape!=(3,)
                or not np.isfinite(self.barycenter).all()):
            raise ValueError('invalid inertial trajectory data')
        if np.any(np.linalg.norm(self.velocities,axis=1)>=1):raise ValueError('inertial velocities must be subluminal')

    def state(self,time):
        v=np.asarray(self.velocities)
        return np.asarray(self.centers)+(time-self.epoch)*v,v,np.zeros((2,3))

    def description(self):
        return dict(model='inertial_pair_control',centers=np.asarray(self.centers).tolist(),
                    velocities=np.asarray(self.velocities).tolist(),epoch=self.epoch,barycenter=list(self.barycenter))
