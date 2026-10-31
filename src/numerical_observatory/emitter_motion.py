"""Circular emitter four-velocities with isolated Schwarzschild and Kerr dynamics."""
from dataclasses import dataclass

import numpy as np

from .metric import Schwarzschild


@dataclass(frozen=True)
class SchwarzschildCircular:
    metric: Schwarzschild
    normal: tuple
    rotation: int = 1

    def __post_init__(self):
        if type(self.metric) is not Schwarzschild:
            raise ValueError("this circular-motion model requires the exact Schwarzschild metric")
        if self.rotation not in (-1, 1):
            raise ValueError("rotation must be +1 or -1")
        n = np.asarray(self.normal, dtype=float)
        if n.shape != (3,) or not np.isfinite(n).all() or np.linalg.norm(n) == 0:
            raise ValueError("normal must be a finite nonzero three-vector")
        object.__setattr__(self, "normal", tuple(n/np.linalg.norm(n)))

    def four_velocity(self, event):
        x = np.asarray(event)[1:4]
        radius = np.linalg.norm(x)
        if radius <= 3*self.metric.mass:
            raise ValueError("timelike Schwarzschild circular geodesics require r>3M")
        if abs(np.dot(self.normal, x)) > 1e-7*radius:
            raise ValueError("emitter event is not in the orbital plane")
        ut = 1/np.sqrt(1-3*self.metric.mass/radius)
        omega = self.rotation*np.sqrt(self.metric.mass/radius**3)
        # dr/dtau=0: outgoing Kerr-Schild time derivative equals Schwarzschild's.
        return ut*np.r_[1.0, omega*np.cross(self.normal, x)]


def kerr_orbit_sign(metric, orbit='prograde'):
    """Coordinate rotation sign; prograde follows J, with +phi at zero spin."""
    from .kerr import Kerr
    if type(metric) is not Kerr or orbit not in ('prograde','retrograde'):
        raise ValueError('Kerr circular orbit must be prograde or retrograde')
    aligned = -1 if metric.spin < 0 else 1
    return aligned if orbit == 'prograde' else -aligned


def kerr_isco(metric, orbit='prograde'):
    """Exact equatorial stable-orbit inner boundary, in coordinate length units.

    Bardeen, Press & Teukolsky (1972). The relative prograde/retrograde branch
    depends on alignment, not on the sign of coordinate angular velocity.
    """
    kerr_orbit_sign(metric,orbit)
    b=abs(metric.spin)
    z1=1+np.cbrt(1-b*b)*(np.cbrt(1+b)+np.cbrt(1-b))
    z2=np.sqrt(3*b*b+z1*z1)
    if b < 1e-3:
        # Exact identity: 3-Z1 = b²(3Z1-1)/Z1². Avoid cancellation
        # that otherwise rounds the linear-in-spin ISCO shift to zero.
        root=b*np.sqrt((3*z1-1)*(3+z1+2*z2))/z1
    else:
        root=np.sqrt(max(0.,(3-z1)*(3+z1+2*z2)))
    return metric.mass*(3+z2+(-root if orbit=='prograde' else root))


@dataclass(frozen=True)
class KerrCircular:
    """Stable equatorial circular emitters in outgoing Cartesian KS coordinates.

    At fixed r,theta the BL -> KS time and azimuth shifts are constant. Thus
    k=d/dT and m=-y d/dx+x d/dy are the same stationary/axial Killing vectors;
    u=ut*(k+Omega*m) is a complete coordinate-transformed four-velocity.
    """
    metric: object
    orbit: str = 'prograde'

    def __post_init__(self):
        kerr_orbit_sign(self.metric,self.orbit)

    @property
    def rotation(self):
        return kerr_orbit_sign(self.metric,self.orbit)

    def omega(self, radius):
        s=self.rotation; M=self.metric.mass
        if not np.isfinite(radius) or radius < kerr_isco(self.metric,self.orbit)*(1-1e-12):
            raise ValueError('Kerr circular disk emitter must be at or outside its ISCO')
        return s*np.sqrt(M)/(radius**1.5+s*self.metric.a*np.sqrt(M))

    def four_velocity(self,event):
        r=self.metric.radius(event)
        if abs(event[3])>1e-7*r:
            raise ValueError('Kerr circular disk emitter must be equatorial (z=0)')
        omega=self.omega(r)
        tangent=np.array([1.,-omega*event[2],omega*event[1],0.])
        norm=-(tangent @ self.metric.covariant(event) @ tangent)
        if not np.isfinite(norm) or norm<=0:
            raise ValueError('Kerr circular emitter is not timelike')
        return tangent/np.sqrt(norm)
