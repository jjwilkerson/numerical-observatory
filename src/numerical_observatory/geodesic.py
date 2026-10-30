"""Canonical null-geodesic integration, events, and sampled constraint diagnostics."""
from dataclasses import dataclass

import numpy as np
from scipy.integrate import quad, solve_ivp

from .metric import Metric, Schwarzschild


@dataclass(frozen=True)
class Integration:
    rtol: float = 1e-8
    atol: float = 1e-10
    escape_radius: float = 100.0
    capture_radius: float = 1.99
    max_affine: float = 500.0
    max_step: float = 5.0
    null_tolerance: float = 1e-5

    def __post_init__(self):
        if any(not np.isfinite(v) or v <= 0 for v in vars(self).values()):
            raise ValueError("integration parameters must be finite and positive")


@dataclass
class RayResult:
    status: str
    direction: np.ndarray | None
    max_null_error: float
    energy_drift: float
    angular_momentum_drift: float
    steps: int
    evaluations: int
    reason: str


def hamilton_rhs(metric: Metric, state):
    event, p = state[:4], state[4:]
    return np.r_[metric.inverse(event) @ p,
                 -0.5 * np.einsum("aij,i,j->a", metric.inverse_derivative(event), p, p)]


def asymptotic_direction(metric: Schwarzschild, state):
    """Continue the outgoing spatial orbit from escape sphere to null infinity.

    In its fixed orbital plane, dphi/du = b/sqrt(1-b²u²+2Mb²u³).
    This Schwarzschild-specific tail removes finite escape-radius direction bias.
    """
    x, p = state[1:4], state[4:]
    r = np.linalg.norm(x)
    radial = x / r
    angular = np.cross(x, p[1:])
    length = np.linalg.norm(angular)
    if length < 1e-13 * metric.mass:
        return radial
    b = length / abs(p[0])
    angle = quad(lambda u: b / np.sqrt(1 - b*b*u*u + 2*metric.mass*b*b*u**3),
                 0, 1/r, epsabs=1e-12, epsrel=1e-12)[0]
    tangential = np.cross(angular / length, radial)
    direction = np.cos(angle) * radial + np.sin(angle) * tangential
    return direction / np.linalg.norm(direction)


def trace(metric: Schwarzschild, event, momentum, settings: Integration):
    """Integrate toward the past with increasing affine parameter.

    The ODE uses only Metric; spherical boundaries and the tail currently
    implement the Schwarzschild observing problem and are intentionally explicit.
    """
    radius = np.linalg.norm(event[1:])
    if not 0 < settings.capture_radius < 2 * metric.mass:
        raise ValueError("capture radius must be inside the horizon and above zero")
    if not settings.escape_radius > max(radius, 3 * metric.mass):
        raise ValueError("escape radius must exceed the camera radius and 3M")
    if radius <= 2 * metric.mass:
        raise ValueError("ray must start outside the horizon")

    def capture(_, state):
        return np.linalg.norm(state[1:4]) - settings.capture_radius

    def escape(_, state):
        return np.linalg.norm(state[1:4]) - settings.escape_radius

    capture.terminal, capture.direction = True, -1
    escape.terminal, escape.direction = True, 1
    sol = solve_ivp(lambda _, y: hamilton_rhs(metric, y), (0, settings.max_affine),
                    np.r_[event, momentum], method="DOP853", rtol=settings.rtol,
                    atol=settings.atol, max_step=settings.max_step,
                    events=(capture, escape), dense_output=True)
    # Accepted endpoints AND interval midpoints; this is a sampled diagnostic,
    # not a rigorous bound between samples. Never project p back onto the cone.
    times = np.sort(np.r_[sol.t, (sol.t[1:] + sol.t[:-1]) / 2])
    samples = sol.sol(times).T
    p = samples[:, 4:]
    inverse = np.array([metric.inverse(y[:4]) for y in samples])
    energy_scale = momentum[0] ** 2
    null = np.abs(np.einsum("ni,nij,nj->n", p, inverse, p)) / energy_scale
    angular = np.cross(samples[:, 1:4], p[:, 1:])
    angular_scale = max(np.linalg.norm(angular[0]), metric.mass * abs(momentum[0]))
    null_error = float(np.max(null))
    status, reason, direction = "unresolved", "affine_limit", None
    if not sol.success:
        reason = "solver_failure: " + sol.message
    elif not np.isfinite(null_error) or null_error > settings.null_tolerance:
        reason = "null_constraint_exceeded"
    elif len(sol.t_events[0]):
        status, reason = "captured", "horizon_crossing"
    elif len(sol.t_events[1]):
        status, reason = "escaped", "outward_escape"
        direction = asymptotic_direction(metric, sol.y[:, -1])
    return RayResult(status, direction, null_error,
                     float(np.max(np.abs(p[:, 0] / momentum[0] - 1))),
                     float(np.max(np.linalg.norm(angular - angular[0], axis=1)) / angular_scale),
                     len(sol.t) - 1, sol.nfev, reason)
