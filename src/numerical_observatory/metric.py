"""Event-based metric contract and outgoing Cartesian Kerr–Schild geometry."""
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]
ETA = np.diag([-1.0, 1.0, 1.0, 1.0])


class Metric(Protocol):
    """All inputs are events (t, x, y, z), including for stationary metrics.

    inverse_derivative[a, mu, nu] = partial_a g^{mu nu}.
    Implementations must provide all four derivatives for time-dependent data.
    """

    def covariant(self, event: Array) -> Array: ...
    def inverse(self, event: Array) -> Array: ...
    def inverse_derivative(self, event: Array) -> Array: ...


@dataclass(frozen=True)
class Schwarzschild:
    mass: float = 1.0

    def __post_init__(self):
        if not np.isfinite(self.mass) or self.mass <= 0:
            raise ValueError("mass must be finite and positive")

    def _geometry(self, event):
        r = np.linalg.norm(event[1:])
        if r <= 0:
            raise ValueError("Schwarzschild curvature singularity at r=0")
        return r, event[1:] / r, 2 * self.mass / r

    def covariant(self, event):
        _, n, f = self._geometry(event)
        ell = np.r_[1.0, -n]
        return ETA + f * np.outer(ell, ell)

    def inverse(self, event):
        _, n, f = self._geometry(event)
        k = np.r_[-1.0, -n]
        return ETA - f * np.outer(k, k)

    def inverse_derivative(self, event):
        r, n, f = self._geometry(event)
        k = np.r_[-1.0, -n]
        dk = np.zeros((3, 4))
        dk[:, 1:] = -(np.eye(3) - np.outer(n, n)) / r
        derivative = np.zeros((4, 4, 4))
        derivative[1:] = (f / r * n[:, None, None] * np.outer(k, k)
                          - f * (dk[:, :, None] * k[None, None, :]
                                 + k[None, :, None] * dk[:, None, :]))
        return derivative

    @property
    def critical_impact_parameter(self):
        return 3 * np.sqrt(3) * self.mass

    def shadow_angle(self, radius):
        """Static observer, r > 3M; inward sky angle in radians."""
        if radius <= 3 * self.mass:
            raise ValueError("shadow-angle formula here requires r > 3M")
        return np.arcsin(self.critical_impact_parameter
                         * np.sqrt(1 - 2 * self.mass / radius) / radius)
