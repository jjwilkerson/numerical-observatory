"""Unwrapped Schwarzschild orbital angles, distinct from disk azimuth.

Wraps the validated orbit() quadrature; no new geodesic solver is introduced.
Outgoing Cartesian Kerr--Schild coordinates have unchanged Schwarzschild r,
theta,phi, with r=norm(x); only the time coordinate differs. In the conserved
orbital plane, dPhi/dlambda=|L|/r**2 is nonnegative along the backward ray.
"""
import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq

from .metric import Schwarzschild
from .schwarzschild_orbit import orbit


class AmbiguousAngle(ValueError):
    """Saved positions do not distinguish two nonidentical orbital branches."""


def orbital_frame(metric, event, momentum):
    if type(metric) is not Schwarzschild:
        raise ValueError("orbital-angle diagnostic requires the exact Schwarzschild metric")
    x = np.asarray(event)[1:4]
    radius = np.linalg.norm(x)
    energy = float(momentum[0])  # positive p_t for past-directed photons
    angular = np.cross(x, np.asarray(momentum)[1:4])
    length = np.linalg.norm(angular)
    if not np.isfinite(np.r_[radius, energy, length]).all() or radius <= 3*metric.mass or energy <= 0:
        raise ValueError("requires a past-directed photon launched outside 3M")
    radial = x/radius
    transverse = None if length == 0 else np.cross(angular/length, radial)
    speed = float(np.dot(radial, (metric.inverse(event) @ momentum)[1:]))
    return radius, length/energy, speed, radial, transverse


class OrbitalAngle:
    """Angle quadratures in units M=1 internally; Phi=0 at the camera."""
    def __init__(self, mass, radius, impact, inward=True, eps=1e-11):
        if not np.isfinite([mass, radius, impact, eps]).all() or mass <= 0 or radius <= 3*mass or impact < 0 or eps <= 0:
            raise ValueError("invalid Schwarzschild angle quadrature parameters")
        self.mass, self.radius, self.b, self.inward, self.eps = mass, radius/mass, impact/mass, inward, eps
        if self.b*np.sqrt(1-2/self.radius)/self.radius >= 1:
            raise ValueError("initial impact parameter is outside the non-tangent launch domain")
        self.turn = self.phase_turn = self.background = None
        self.error_estimate = 0.
        critical = 3*np.sqrt(3)
        self.critical_uncertain = abs(self.b/critical-1) <= 64*np.finfo(float).eps
        if self.b == 0:
            self.outcome = "captured" if inward else "escaped"
            self.background = None if inward else 0.
        elif not inward:
            self.outcome = "escaped"
            self.background = self._u_integral(0., 1/self.radius)
        elif self.critical_uncertain:
            self.outcome = "unresolved"
        elif self.b < critical:
            self.outcome = "captured"
        else:
            angle = np.arcsin(self.b*np.sqrt(1-2/self.radius)/self.radius)
            solved = orbit(angle, self.radius, epsabs=eps, epsrel=eps)
            self.turn = solved.turning_radius
            self.background = solved.phase
            self.error_estimate = solved.error_estimate
            self.phase_turn = self._s_integral(0, np.sqrt(1-self.turn/self.radius))
            self.outcome = "escaped"

    @classmethod
    def from_initial(cls, metric, event, momentum, eps=1e-11):
        radius, impact, speed, _, _ = orbital_frame(metric, event, momentum)
        if abs(speed) <= 1e-13*abs(momentum[0]):
            raise ValueError("exactly tangent launch requires a separate endpoint limit")
        return cls(metric.mass, radius, impact, speed < 0, eps)

    def _s_integrand(self, s):
        u0 = 1/self.turn
        u = u0*(1-s*s)
        # Same stable positive factorization as schwarzschild_orbit.orbit.
        factor = (u0+u)*((self.turn-3)/self.turn)+(u0-u)*(u0+2*u)
        return 2*np.sqrt(u0)/np.sqrt(factor)

    def _s_integral(self, low, high):
        return quad(self._s_integrand, low, high, epsabs=self.eps, epsrel=self.eps, limit=200)[0]

    def _u_integral(self, low, high):
        if self.b == 0 or low == high:
            return 0.
        def integrand(u):
            radicand = 1-self.b**2*u*u+2*self.b**2*u**3
            if radicand <= 0:
                raise ValueError("radius crosses a forbidden branch or unresolved critical orbit")
            return self.b/np.sqrt(radicand)
        return quad(integrand, low, high, epsabs=self.eps, epsrel=self.eps, limit=200)[0]

    def to_radius(self, radius, branch="inward"):
        """Continuous angle on the specified leg; no modulo or crossing count."""
        r = float(radius)/self.mass
        if not np.isfinite(r) or r <= 0:
            raise ValueError("hit radius must be positive and finite")
        if branch not in ("inward", "outward"):
            raise ValueError("branch must be inward or outward")
        if not self.inward:
            if branch != "outward" or r < self.radius:
                raise ValueError("outward launch cannot reach this radius on that leg")
            return self._u_integral(1/r, 1/self.radius)
        if self.turn is None:
            if branch != "inward" or r > self.radius:
                raise ValueError("captured/critical inward ray has no outward radial leg")
            if self.critical_uncertain and r <= 3:
                raise ValueError("critical ray does not reach the photon sphere in finite affine time")
            return self._u_integral(1/self.radius, 1/r)
        if r < self.turn or (branch == "inward" and r > self.radius):
            raise ValueError("hit radius is not on this trajectory branch")
        s = np.sqrt(1-self.turn/r)
        s_camera = np.sqrt(1-self.turn/self.radius)
        if r == self.turn:
            return self.phase_turn
        if branch == "inward":
            # Integrate this interval directly instead of subtracting two large phases.
            return self._s_integral(s, s_camera)
        return self.phase_turn+self._s_integral(0., s)

    def saved_hit(self, position, radial, transverse, match_tolerance=5e-6):
        """Select a radial leg using actual position in the INITIAL orbital plane.

        Both candidates come from independent radial quadrature. Near a spatial
        self-intersection the saved position can be insufficient: report that
        ambiguity, never use the disk crossing index to choose an angle.
        """
        position = np.asarray(position, dtype=float)
        radius = np.linalg.norm(position)
        if position.shape != (3,) or not np.isfinite(position).all() or radius == 0:
            raise ValueError("invalid saved hit position")
        unit = position/radius
        branches = ["inward", "outward"] if self.turn is not None else ["inward" if self.inward else "outward"]
        candidates = []
        for branch in branches:
            try:
                angle = self.to_radius(radius, branch)
            except ValueError:
                continue
            predicted = radial if transverse is None else np.cos(angle)*radial+np.sin(angle)*transverse
            error = float(np.arctan2(np.linalg.norm(np.cross(predicted, unit)), np.dot(predicted, unit)))
            if error <= match_tolerance:
                candidates.append((angle, branch, error))
        if not candidates:
            raise ValueError("saved position does not match either conserved-quantity trajectory branch")
        if len(candidates) == 2 and abs(candidates[0][0]-candidates[1][0]) > 32*np.finfo(float).eps*max(1, candidates[0][0]):
            raise AmbiguousAngle("both radial legs match the saved position; momentum or optional ODE audit is needed")
        return min(candidates, key=lambda candidate: candidate[2])


def alignment_rings(metric, camera, orders=(1, 2, 3, 4, 5), tolerances=(1e-8, 1e-10, 1e-12)):
    from .schwarzschild_orbit import compatible
    if not compatible(metric, camera):
        raise ValueError("circular image-plane ring overlays require a centered static Schwarzschild camera")
    radius = np.linalg.norm(camera.position)
    width = getattr(camera, "reference_width", camera.width)
    factor = width/(2*np.tan(np.deg2rad(camera.fov_degrees)/2))
    critical_angle = metric.shadow_angle(radius)
    shadow_pixel = float(factor*np.tan(critical_angle))
    critical = metric.critical_impact_parameter
    maximum = radius/np.sqrt(1-2*metric.mass/radius)
    rings = []
    for order in orders:
        runs = []
        for eps in tolerances:
            def residual(b):
                angle = np.arcsin(b*np.sqrt(1-2*metric.mass/radius)/radius)
                return orbit(angle, radius/metric.mass, epsabs=eps, epsrel=eps).phase-order*np.pi
            low, high = critical*(1+1e-11), maximum*(1-1e-12)
            impact = brentq(residual, low, high, xtol=5e-15, rtol=4*np.finfo(float).eps)
            angle = np.arcsin(impact*np.sqrt(1-2*metric.mass/radius)/radius)
            runs.append(dict(eps=eps, impact_parameter_M=float(impact/metric.mass),
                             angle_degrees=float(np.rad2deg(angle)), radius_pixels=float(factor*np.tan(angle)),
                             residual_radians=float(residual(impact)), bracket_impact_M=[low/metric.mass, high/metric.mass]))
        rings.append(dict(phase_pi=order, **{k: v for k, v in runs[-1].items() if k != "eps"},
                          separation_from_shadow_pixels=runs[-1]["radius_pixels"]-shadow_pixel,
                          convergence=runs,
                          radius_convergence_span_pixels=max(v["radius_pixels"] for v in runs)-min(v["radius_pixels"] for v in runs)))
    center = [(width-1)/2-getattr(camera, "origin_x", 0),
              (getattr(camera, "reference_height", camera.height)-1)/2-getattr(camera, "origin_y", 0)]
    return dict(camera_radius_M=float(radius/metric.mass), full_horizontal_fov_degrees=camera.fov_degrees,
                center_xy=center, pixels_per_tan_angle=float(factor),
                shadow=dict(impact_parameter_M=float(critical/metric.mass), angle_degrees=float(np.rad2deg(critical_angle)), radius_pixels=shadow_pixel),
                rings=rings, background="null infinity; same outgoing analytic tail as existing celestial renderer")
