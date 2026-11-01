"""Time-dependent boosted nonspinning KS superposition, not a vacuum binary."""
from dataclasses import dataclass,field
import numpy as np
from .metric import ETA
from .trajectories import CircularOrbit
from .boosted_ks import contribution,rest_position


@dataclass(frozen=True)
class BinaryKSOrbitApprox:
    masses: tuple = (.5,.5)
    trajectory: object = field(default_factory=CircularOrbit)

    def __post_init__(self):
        m=np.asarray(self.masses,dtype=float)
        if m.shape!=(2,) or not np.isfinite(m).all() or np.any(m<0) or not 0<sum(m)<np.inf:
            raise ValueError('boosted masses must be finite, nonnegative, with finite positive total')
        if isinstance(self.trajectory,CircularOrbit) and tuple(self.trajectory.masses)!=tuple(m):
            raise ValueError('trajectory and metric masses must match')
        object.__setattr__(self,'masses',tuple(m))
        self.body_states(self.trajectory.epoch)

    @property
    def mass(self):return sum(self.masses)

    @property
    def barycenter(self):return np.asarray(self.trajectory.barycenter)

    def body_states(self,time):
        states=self.trajectory.state(time)
        if any(np.asarray(v).shape!=(2,3) or not np.isfinite(v).all() for v in states):
            raise ValueError('invalid trajectory position/velocity/acceleration')
        if not np.iscomplexobj(time) and np.any(np.sum(np.asarray(states[1])**2,axis=1)>=1):
            raise ValueError('superluminal trajectory velocity')
        return states

    def geometry(self,event,derivatives=False):
        event=np.asarray(event)
        if event.shape!=(4,) or not np.isfinite(event).all():raise ValueError('expected finite (t,x,y,z)')
        positions,velocities,accelerations=self.body_states(event[0])
        g=ETA.astype(np.result_type(event,float));dg=np.zeros((4,4,4),dtype=g.dtype)
        for i,m in enumerate(self.masses):
            if m==0:continue
            result=contribution(event,m,positions[i],velocities[i],accelerations[i],derivatives)
            if derivatives:g+=result[0];dg+=result[1]
            else:g+=result
        return (g,dg) if derivatives else g

    def covariant(self,event):return self.geometry(event)
    def covariant_derivative(self,event):return self.geometry(event,True)[1]
    def inverse(self,event):return np.linalg.inv(self.covariant(event))

    def inverse_derivative(self,event):
        g,dg=self.geometry(event,True);inverse=np.linalg.inv(g)
        return -np.einsum('ij,ajk,kl->ail',inverse,dg,inverse)

    def signature_margin(self,event):
        # Spatial block remains SPD (positive-mass outer products). With this
        # fact, -det(g)>0 establishes one negative eigenvalue. Do not reuse the
        # stationary n1.n2 expression for boosted covectors.
        return float(-np.linalg.det(self.covariant(event)))

    def rest_radii(self,event):
        s,v,_=self.body_states(event[0])
        return np.array([np.linalg.norm(rest_position(event,s[i],v[i])) if m else np.inf for i,m in enumerate(self.masses)])

    def helical_quantity(self,states):
        omega=self.trajectory.helical_omega
        if omega is None:return None
        z=np.asarray(states);x=z[...,1]-self.barycenter[0];y=z[...,2]-self.barycenter[1]
        return z[...,4]+omega*(x*z[...,6]-y*z[...,5])

    def description(self):
        return dict(type='binary_ks_orbit_approx',experimental=True,exact_vacuum_solution=False,
                    masses=list(self.masses),total_mass_parameter=self.mass,trajectory=self.trajectory.description(),
                    coordinates='outgoing Cartesian KS, instantaneous Lorentz/no-acceleration-Jacobian ansatz',
                    derivative_policy='analytic derivatives of the full prescribed metric, including velocity and acceleration',
                    warning='Not the published PN/merger metric; moving capture ellipsoids are provisional, not computed horizons.')
