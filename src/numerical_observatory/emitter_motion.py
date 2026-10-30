"""Emitter four-velocities; Schwarzschild assumptions are confined here."""
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
