"""Canonical null-geodesic integration, events, and sampled constraint diagnostics."""
from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.integrate import quad, solve_ivp

from .metric import Metric, Schwarzschild
from .surfaces import CoplanarRay, SurfaceHit


@dataclass(frozen=True)
class Integration:
    rtol: float = 1e-8
    atol: float = 1e-10
    escape_radius: float = 100.0
    capture_radius: float = 1.99  # Schwarzschild only
    max_affine: float = 500.0
    max_step: float = 5.0
    null_tolerance: float = 1e-5
    kerr_capture_fraction: float = 0.995  # Kerr r/r_plus, strictly inside horizon
    kerr_source_radius: float = 100000.0  # Kerr background tail endpoint

    def __post_init__(self):
        if any(not np.isfinite(v) or v <= 0 for v in vars(self).values()):
            raise ValueError("integration parameters must be finite and positive")
        if self.kerr_capture_fraction >= 1:
            raise ValueError("kerr_capture_fraction must be strictly between 0 and 1")


@dataclass
class RayResult:
    status: str
    direction: np.ndarray | None
    max_null_error: float
    energy_drift: float | None
    angular_momentum_drift: float | None
    steps: int
    evaluations: int
    reason: str
    raw_status: str | None = None
    critical_uncertainty: float | None = None
    hit: SurfaceHit | None = None
    carter_drift: float | None = None
    kerr_constants: np.ndarray | None = None


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


@dataclass(frozen=True)
class TraceDomain:
    """Explicit physical/domain policy for a general metric.

    Level sets use negative-going capture and positive-going escape crossings.
    source_direction maps an escaped state to a unit sky direction in the
    background's frame. It must be supplied: no asymptotic flatness is assumed.
    Optional valid_region is positive inside the numerical-data domain.
    """
    capture: Callable
    escape: Callable
    source_direction: Callable
    valid_region: Callable | None = None


def trace(metric: Metric, event, momentum, settings: Integration, domain=None,
          return_solution=False, critical_guard=True, surface=None):
    """Integrate toward the past with increasing affine parameter.

    The ODE uses only Metric. Other geometries require TraceDomain;
    the exact Schwarzschild and Kerr classes have built-in domain policies.
    return_solution exposes dense output for audits. critical_guard=False is
    intended for measuring raw solver error, not ordinary production rendering.
    """
    schwarzschild = type(metric) is Schwarzschild
    from .kerr import Kerr
    kerr_domain = domain is None and type(metric) is Kerr
    builtin_domain = domain is None and schwarzschild
    if kerr_domain:
        if surface is not None:
            raise ValueError("Kerr emitting surfaces are not implemented")
        from .kerr_support import default_domain
        domain = default_domain(metric, event, settings)
    if domain is None:
        if not schwarzschild:
            raise ValueError("non-Schwarzschild metrics require an explicit TraceDomain")
        radius = np.linalg.norm(event[1:])
        if not 0 < settings.capture_radius < 2 * metric.mass:
            raise ValueError("capture radius must be inside the horizon and above zero")
        if not settings.escape_radius > max(radius, 3 * metric.mass):
            raise ValueError("escape radius must exceed the camera radius and 3M")
        if radius <= 2 * metric.mass:
            raise ValueError("ray must start outside the horizon")
        domain = TraceDomain(lambda x: np.linalg.norm(x[1:])-settings.capture_radius,
                             lambda x: np.linalg.norm(x[1:])-settings.escape_radius,
                             lambda y: asymptotic_direction(metric, y))

    def capture(_, state):
        return domain.capture(state[:4])

    def escape(_, state):
        return domain.escape(state[:4])

    capture.terminal, capture.direction = True, -1
    escape.terminal, escape.direction = True, 1
    events = [capture, escape]
    if domain.valid_region is not None:
        def data_boundary(_, state):
            return domain.valid_region(state[:4])
        data_boundary.terminal, data_boundary.direction = True, -1
        events.append(data_boundary)
    sol = solve_ivp(lambda _, y: hamilton_rhs(metric, y), (0, settings.max_affine),
                    np.r_[event, momentum], method="DOP853", rtol=settings.rtol,
                    atol=settings.atol, max_step=settings.max_step,
                    events=events, dense_output=True)
    result = summarize_solution(metric, event, momentum, settings, sol, domain,
                                builtin_domain, critical_guard, surface)
    if type(metric) is Kerr:
        from .kerr_support import finish_kerr
        result = finish_kerr(metric, event, momentum, settings, sol, result, kerr_domain)
    if return_solution and result.hit is not None:
        # The public opaque-ray solution terminates at the emitting surface.
        # Only geometry caches (which call without a surface) retain later data.
        original = sol
        stop = result.hit.affine
        sol = type(original)(original)
        sol.t = np.r_[original.t[original.t < stop], stop]
        sol.y = original.sol(sol.t)
        sol.t_events = [t[t <= stop] for t in original.t_events]+[np.array([stop])]
        sol.y_events = [original.sol(t).T if len(t) else np.empty((0, 8))
                        for t in sol.t_events]
        def terminated_dense(t):
            if np.any(np.asarray(t) > stop+1e-10):
                raise ValueError("opaque ray terminates at the first emitting surface")
            return original.sol(t)
        sol.sol = terminated_dense
        sol.success, sol.status, sol.message = True, 1, "First opaque surface reached"
    return (result, sol) if return_solution else result


def summarize_solution(metric, event, momentum, settings, sol, domain=None,
                       builtin_domain=True, critical_guard=True, surface=None):
    """First-hit termination and diagnostics on a precomputed dense trajectory.

    Dense solutions can be reused under exact Schwarzschild rotations. The
    opaque ray ends at its first valid surface hit; later solver data never
    contributes emission or constraint checks. Retaining that later geometry
    is a computational convenience, not transmission through an opaque surface.
    """
    schwarzschild = type(metric) is Schwarzschild
    radius = np.linalg.norm(event[1:])
    hit, coplanar = None, False
    if surface is not None:
        try:
            hit = surface.first_hit(sol)
        except CoplanarRay:
            coplanar = True
    stop = sol.t[-1] if hit is None else hit.affine
    accepted = np.r_[sol.t[sol.t < stop], stop]
    # Accepted endpoints AND interval midpoints; this is a sampled diagnostic,
    # not a rigorous bound between samples. Never project p back onto the cone.
    times = np.sort(np.r_[accepted, (accepted[1:] + accepted[:-1]) / 2])
    samples = sol.sol(times).T
    p = samples[:, 4:]
    inverse = np.array([metric.inverse(y[:4]) for y in samples])
    energy_scale = momentum[0] ** 2 if abs(momentum[0]) > 1e-14 else np.dot(momentum, momentum)
    null = np.abs(np.einsum("ni,nij,nj->n", p, inverse, p)) / energy_scale
    angular = np.cross(samples[:, 1:4], p[:, 1:])
    angular_scale = max(np.linalg.norm(angular[0]), metric.mass * abs(momentum[0])) if schwarzschild else 1
    null_error = float(np.max(null))
    status, reason, direction = "unresolved", "affine_limit", None
    if coplanar:
        reason = "coplanar_surface"
    elif not sol.success and hit is None:
        reason = "solver_failure: " + sol.message
    elif not np.isfinite(null_error) or null_error > settings.null_tolerance:
        reason = "null_constraint_exceeded"
    elif hit is not None:
        status, reason = "emitted", "first_opaque_surface"
    elif len(sol.t_events[0]):
        status, reason = "captured", "capture_boundary"
    elif len(sol.t_events[1]):
        status, reason = "escaped", "outward_escape"
        direction = np.asarray(asymptotic_direction(metric, sol.y[:, -1]) if domain is None
                               else domain.source_direction(sol.y[:, -1]), dtype=float)
        if direction.shape != (3,) or not np.isfinite(direction).all() or np.linalg.norm(direction) == 0:
            raise ValueError("source_direction must return a finite, nonzero three-vector")
        direction /= np.linalg.norm(direction)
    elif len(sol.t_events) == 3 and len(sol.t_events[2]):
        reason = "data_domain_exit"
    raw_status = status
    critical_uncertainty = None
    if builtin_domain and radius > 3*metric.mass:
        initial_velocity = metric.inverse(event) @ momentum
        if np.dot(event[1:], initial_velocity[1:]) < 0:
            impact = np.linalg.norm(angular, axis=1)/np.abs(p[:, 0])
            critical_uncertainty = float(max(np.max(np.abs(impact-impact[0]))/metric.critical_impact_parameter,
                                             null_error, 64*np.finfo(float).eps))
            margin = abs(impact[0]/metric.critical_impact_parameter-1)
            # An empirical reliability guard, not a rigorous global error bound.
            # A small null residual alone cannot certify a near-critical outcome.
            if critical_guard and margin <= critical_uncertainty and status in ("captured", "escaped"):
                status, reason, direction = "unresolved", "near_critical_uncertainty", None
    result = RayResult(status, direction, null_error,
                     float(np.max(np.abs(p[:, 0] / momentum[0] - 1))) if schwarzschild else None,
                     float(np.max(np.linalg.norm(angular - angular[0], axis=1)) / angular_scale) if schwarzschild else None,
                     len(sol.t) - 1, sol.nfev, reason, raw_status, critical_uncertainty,
                     hit if status == "emitted" else None)
    return result
