"""Optional Schwarzschild-only orbit quadrature; never a general metric backend."""
from dataclasses import dataclass

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq

from .metric import Schwarzschild


@dataclass(frozen=True)
class Orbit:
    phase: float
    turning_radius: float
    error_estimate: float

    @property
    def direction(self):
        return np.array([np.sin(self.phase), 0.0, np.cos(self.phase)])


def orbit(angle, radius=20.0, epsabs=1e-11, epsrel=1e-11):
    """Inward escaping orbit, M=1, observer on +z; phase is NOT modulo 2pi.

    The substitution u=u_turn*(1-s²) removes the turning-point singularity.
    A positive-term factorization avoids cancellation near the photon sphere.
    Very near b_crit double precision itself limits recoverable deflection.
    """
    if radius <= 3 or not 0 < angle < np.pi/2:
        raise ValueError("orbit quadrature requires r>3M and an inward nonradial ray")
    b = radius*np.sin(angle)/np.sqrt(1-2/radius)
    relative = b/(3*np.sqrt(3)) - 1
    if relative <= 0:
        raise ValueError("orbit quadrature requires b>b_crit")
    excess = 27*relative*(relative+2)
    turning = brentq(lambda r: (r-3)**2*(r+6)-excess*(r-2),
                     3.0, radius, xtol=5e-15, rtol=4*np.finfo(float).eps)
    u0 = 1/turning
    def integrand(s):
        u = u0*(1-s*s)
        factor = (u0+u)*((turning-3)/turning)+(u0-u)*(u0+2*u)
        return 2*np.sqrt(u0)/np.sqrt(factor)
    a, ea = quad(integrand, 0, np.sqrt(1-turning/radius), epsabs=epsabs, epsrel=epsrel, limit=200)
    b, eb = quad(integrand, 0, 1, epsabs=epsabs, epsrel=epsrel, limit=200)
    return Orbit(a+b, turning, ea+eb)


def compatible(metric, camera):
    radius = np.linalg.norm(camera.position)
    return bool(type(metric) is Schwarzschild and camera.observer_frame is None
                and radius > 3*metric.mass
                and np.linalg.norm(camera.axes()[0]+np.asarray(camera.position)/radius) < 1e-14)


def ray_map(metric, camera, samples=1, window=None):
    """Subpixel ray directions with exact rotational reuse; explicit guard.

    Each subpixel follows its own orbit; there is no direction interpolation.
    Near machine-precision-critical rays are unresolved rather than captured.
    """
    if not compatible(metric, camera):
        raise ValueError("Schwarzschild quadrature is incompatible with this metric or camera; use the ODE tracer")
    if not isinstance(samples, int) or samples < 1:
        raise ValueError("samples must be a positive integer")
    forward, right, up = camera.axes()
    radius = np.linalg.norm(camera.position)
    h, w = camera.height*samples, camera.width*samples
    if window is None:
        yy, xx = np.mgrid[:h, :w]
    else:
        r0, r1, c0, c1 = window
        if not (0 <= r0 < r1 <= camera.height and 0 <= c0 < c1 <= camera.width):
            raise ValueError("window lies outside camera image")
        yy, xx = np.mgrid[r0*samples:r1*samples, c0*samples:c1*samples]
    ix, iy = 2*xx+1-w, h-2*yy-1
    k = ix*ix+iy*iy
    keys, reverse = np.unique(k, return_inverse=True)
    angles = np.arctan(np.sqrt(keys)*np.tan(np.deg2rad(camera.fov_degrees)/2)/w)
    critical = metric.shadow_angle(radius)
    ambiguous = np.abs(angles-critical) <= 32*np.finfo(float).eps
    escaped = (angles > critical) & ~ambiguous
    orbits = np.full((len(keys), 3), np.nan)
    for i in np.flatnonzero(escaped):
        orbits[i] = orbit(angles[i], radius/metric.mass).direction
    mapped = orbits[reverse].reshape(*k.shape, 3)
    transverse = (ix[..., None]*right+iy[..., None]*up)/np.maximum(np.sqrt(k)[..., None], 1)
    directions = mapped[..., 0, None]*transverse-mapped[..., 2, None]*forward
    codes = np.where(ambiguous, 2, escaped.astype(np.uint8)).astype(np.uint8)
    return directions, codes[reverse].reshape(k.shape), len(keys)
