"""Opt-in plane-crossing collector. Ordinary opaque tracing is untouched."""
from dataclasses import dataclass
import warnings

import numpy as np
from scipy.integrate import quad, IntegrationWarning
from scipy.optimize import brentq

from .camera import Camera
from .disk_diagnostics import disk_basis
from .emission import frequency_shift
from .geodesic import summarize_solution
from .orbital_angle import OrbitalAngle, orbital_frame
from .orbital_validation import pixel_momentum
from .surfaces import SurfaceHit, CoplanarRay
from .surface_render import SurfacePathCache


# One row per actual plane crossing, NOT one row per emitting intersection.
CROSSING_DTYPE = np.dtype([
    ("plane_ordinal", "u4"), ("emitting_ordinal", "u4"),
    ("event", "f8", (4,)), ("momentum", "f8", (4,)),
    ("affine", "f8"), ("radius_M", "f8"), ("azimuth_degrees", "f8"),
    ("region", "u1"), ("geometry_valid", "?"), ("emission_valid", "?"),
    ("g", "f8"), ("bolometric_intensity", "f8"), ("observed_temperature", "f8"),
    ("phi_radians", "f8"), ("null_error", "f8"),
])
REGIONS = {0: "invalid", 1: "central hole", 2: "emitting annulus", 3: "beyond outer edge"}
REASONS = {0: "capture_boundary", 1: "outward_escape", 2: "affine_limit",
           3: "numerical_failure", 4: "coplanar_surface"}


def region(radius, surface):
    if not np.isfinite(radius) or radius <= 0:
        return 0
    return 1 if radius < surface.inner_radius else 3 if radius > surface.outer_radius else 2


def pack_crossings(metric, initial_p, observer, scene, hits, null_errors=None, phases=None):
    """Evaluate existing motion and emission models independently at each hit."""
    output = np.zeros(len(hits), dtype=CROSSING_DTYPE)
    _, e0, e90 = disk_basis(scene.surface.normal)
    count = 0
    for i, hit in enumerate(hits):
        row = output[i]
        radius = float(np.linalg.norm(hit.event[1:4]))
        p = hit.momentum
        residual = float(abs(p @ metric.inverse(hit.event) @ p)/initial_p[0]**2)
        error = residual if null_errors is None else max(residual, null_errors[i])
        valid = np.isfinite(error) and np.isfinite(np.r_[hit.event, p, hit.affine]).all()
        row["plane_ordinal"], row["event"], row["momentum"], row["affine"] = hit.crossing_index, hit.event, p, hit.affine
        row["radius_M"], row["region"], row["null_error"] = radius/metric.mass, region(radius, scene.surface), error
        row["geometry_valid"] = valid
        row["azimuth_degrees"] = np.degrees(np.arctan2(hit.event[1:4] @ e90, hit.event[1:4] @ e0)) % 360
        row["phi_radians"] = np.nan if phases is None else phases[i]
        row["g"], row["observed_temperature"] = np.nan, np.nan
        if valid and row["region"] == 2:
            try:
                g = frequency_shift(initial_p, observer, p, scene.motion.four_velocity(hit.event))
                intensity = scene.emission.observed_bolometric(hit.event, g)
                if not np.isfinite([g, intensity]).all() or intensity < 0:
                    raise ValueError("invalid emission")
            except ValueError:
                continue
            count += 1
            row["emitting_ordinal"], row["emission_valid"] = count, True
            row["g"], row["bolometric_intensity"], row["observed_temperature"] = g, intensity, g*scene.emission.temperature(hit.event)
    return output


def collect_solution(metric, event, momentum, settings, solution, scene, maximum):
    """Same dense ODE solution and plane-root detector as the opaque renderer."""
    try:
        all_hits = scene.surface.crossings(solution)
    except CoplanarRay:
        return np.empty(0, CROSSING_DTYPE), 2, 4, np.nan, np.full(3, np.nan), False
    hits = all_hits[:maximum]
    times = np.sort(np.unique(np.r_[solution.t, (solution.t[1:]+solution.t[:-1])/2,
                                    [hit.affine for hit in hits]]))
    states = solution.sol(times).T
    errors = np.array([abs(y[4:] @ metric.inverse(y[:4]) @ y[4:])/momentum[0]**2 for y in states])
    prefix = np.maximum.accumulate(np.where(np.isfinite(errors), errors, np.inf))
    at_hits = [prefix[np.searchsorted(times, hit.affine)] for hit in hits]
    # Optional unwrapped orbital angle, from this actual dense trajectory.
    try:
        _, _, _, radial, transverse = orbital_frame(metric, event, momentum)
        phases = np.zeros(len(times)) if transverse is None else np.unwrap(np.arctan2(states[:, 1:4] @ transverse, states[:, 1:4] @ radial))
        phases -= phases[0]
        phase_values = [phases[np.searchsorted(times, hit.affine)] for hit in hits]
        if len(phases) > 1 and np.max(abs(np.diff(phases))) >= np.pi/2:
            phase_values = None
    except ValueError:
        phase_values = None
    rows = pack_crossings(metric, momentum, scene.motion_observer, scene, hits, at_hits, phase_values)
    bad = rows["null_error"] > settings.null_tolerance
    rows["geometry_valid"][bad] = False
    rows["emission_valid"][bad] = False
    rows["g"][bad], rows["bolometric_intensity"][bad], rows["observed_temperature"][bad] = np.nan, 0., np.nan
    # Recompute emitting ordinal after rejecting any invalid state.
    rows["emitting_ordinal"] = np.where(rows["emission_valid"], np.cumsum(rows["emission_valid"]), 0)
    end = summarize_solution(metric, event, momentum, settings, solution)
    status = {"captured": 0, "escaped": 1, "unresolved": 2}[end.status]
    reason = 0 if status == 0 else 1 if status == 1 else 2 if end.reason == "affine_limit" else 3
    direction = np.full(3, np.nan) if end.direction is None else end.direction
    return rows, status, reason, end.max_null_error, direction, len(all_hits) > maximum


class QuadratureUnsupported(ValueError):
    pass


def checked_quadrature(function, *args):
    with warnings.catch_warnings():
        warnings.simplefilter("error", IntegrationWarning)
        return function(*args)


class LayerOrbit:
    """A single exact Schwarzschild orbit, reused for rotated photon planes.

    Extends the existing OrbitalAngle/disk_validation quadratures to every
    plane crossing, including captured rays. No direction interpolation.
    """
    def __init__(self, metric, event, momentum, settings):
        self.metric, self.event, self.p, self.settings = metric, event, momentum, settings
        radius, impact, speed, self.radial, self.transverse = orbital_frame(metric, event, momentum)
        if speed >= 0 or radius <= 3*metric.mass:
            raise QuadratureUnsupported("quadrature requires an inward launch outside 3M")
        self.E = float(momentum[0])
        self.L = impact*self.E
        self.angle = OrbitalAngle(metric.mass, radius, impact, True, eps=1e-11)
        if self.angle.outcome == "unresolved":
            raise QuadratureUnsupported("critical orbit requires ODE reliability checks")
        self.outcome = self.angle.outcome
        self.u_camera = 1/self.angle.radius
        self.limit_radius = settings.escape_radius if self.outcome == "escaped" else settings.capture_radius
        if self.L == 0:
            self.phase_end = 0.
            self.end_affine = abs(self.limit_radius-radius)/self.E
        elif self.angle.turn is not None:
            self.s_camera = np.sqrt(1-self.angle.turn/self.angle.radius)
            self.s_end = np.sqrt(1-self.angle.turn/(self.limit_radius/metric.mass))
            self.phase_end = self.angle.to_radius(self.limit_radius, "outward")
            self.end_affine = self.elapsed(0., self.s_camera, -1)[0]+self.elapsed(0., self.s_end, 1)[0]
        else:
            self.u_end = metric.mass/self.limit_radius
            self.phase_end = self.angle.to_radius(self.limit_radius, "inward")
            self.end_affine = self.elapsed(self.u_camera, self.u_end, -1)[0]

    def components(self, radius, sign):
        f = 2*self.metric.mass/radius
        lapse = 1-f
        D = max(0., 1-lapse*(self.L/self.E/radius)**2)
        root = np.sqrt(D)
        if sign < 0:
            # Rationalized formula remains regular across the past horizon.
            pr = self.E*((self.L/self.E/radius)**2-(1+f))/(f+root)
        else:
            pr = self.E*(root+f)/lapse
        dt = -(1+f)*self.E-f*pr
        return pr, dt

    def elapsed(self, low, high, sign):
        if low == high:
            return 0., 0.
        mass = self.metric.mass
        if self.angle.turn is not None:
            def functions(v):
                r = mass*self.angle.turn/(1-v*v)
                rate = self.angle._s_integrand(v)*r*r/self.L
                return rate, rate*self.components(r, sign)[1]
            points = None
        else:
            def functions(u):
                r = mass/u
                D = 1-self.angle.b**2*u*u+2*self.angle.b**2*u**3
                rate = mass/(self.E*u*u*np.sqrt(D))
                return rate, rate*self.components(r, sign)[1]
            points = [1/3] if low < 1/3 < high else None
        affine = quad(lambda v: functions(v)[0], low, high, epsabs=1e-9, epsrel=1e-11, points=points, limit=200)[0]
        time = quad(lambda v: functions(v)[1], low, high, epsabs=1e-9, epsrel=1e-11, points=points, limit=200)[0]
        return affine, time

    def crossing(self, phase, ordinal, transverse):
        if self.angle.turn is not None:
            sign = -1 if phase <= self.angle.phase_turn else 1
            target = abs(phase-self.angle.phase_turn)
            upper = self.s_camera if sign < 0 else self.s_end
            s = 0. if target == 0 else brentq(lambda s: self.angle._s_integral(0, s)-target, 0, upper, xtol=1e-13, rtol=1e-13)
            radius = self.metric.mass*self.angle.turn/(1-s*s)
            if sign < 0:
                affine, time = self.elapsed(s, self.s_camera, -1)
            else:
                ai, ti = self.elapsed(0, self.s_camera, -1)
                ao, to = self.elapsed(0, s, 1)
                affine, time = ai+ao, ti+to
        else:
            u = brentq(lambda u: self.angle._u_integral(self.u_camera, u)-phase,
                       self.u_camera, self.u_end, xtol=1e-14, rtol=1e-13)
            radius = self.metric.mass/u
            sign = -1
            affine, time = self.elapsed(self.u_camera, u, -1)
        n = np.cos(phase)*self.radial+np.sin(phase)*transverse
        tangent = -np.sin(phase)*self.radial+np.cos(phase)*transverse
        pr, _ = self.components(radius, sign)
        p = np.r_[self.E, pr*n+self.L/radius*tangent]
        return SurfaceHit(float(affine), np.r_[self.event[0]+time, radius*n], p, ordinal)

    def crossings(self, surface, transverse, maximum):
        a = np.dot(surface.normal, self.radial)
        b = 0. if transverse is None else np.dot(surface.normal, transverse)
        if np.hypot(a, b) < 1e-12:
            raise CoplanarRay("coplanar ray has no isolated intersections")
        if self.L == 0:
            return [], [], False
        first = np.arctan2(-a, b) % np.pi
        if first < 1e-12:
            first += np.pi
        total = max(0, int(np.floor((self.phase_end-first)/np.pi))+1)
        hits, phases = [], []
        for i in range(min(maximum, total)):
            phase = first+i*np.pi
            hit = self.crossing(phase, i+1, transverse)
            if hit.affine > self.settings.max_affine:
                break
            hits.append(hit); phases.append(phase)
        more = total > maximum and len(hits) == maximum
        return hits, phases, more


@dataclass
class LayerScene:
    surface: object
    motion: object
    emission: object
    display: object
    motion_observer: object


def reference_camera(camera):
    """Full virtual camera for exact reuse even when processing a window."""
    if not hasattr(camera, "reference_width"):
        return camera
    return Camera(position=camera.position, target=camera.target, up=camera.up, time=camera.time,
                  fov_degrees=camera.fov_degrees, observer_frame=camera.observer_frame,
                  width=camera.reference_width, height=camera.reference_height)


def layer_task(spec, indices):
    metric, camera, settings, disk, maximum, method = spec
    scene = LayerScene(disk.surface, disk.motion, disk.emission, disk.display, camera.tetrad(metric)[0])
    full = reference_camera(camera)
    cache = SurfacePathCache(metric, full, settings)
    orbits = {}
    forward, right, up = full.axes()
    values = []
    for index in indices:
        row, col = divmod(int(index), camera.width)
        global_row, global_col = row+getattr(camera, "origin_y", 0), col+getattr(camera, "origin_x", 0)
        ix, iy = 2*global_col+1-full.width, full.height-2*global_row-1
        key = ix*ix+iy*iy
        p = pixel_momentum(camera, metric, col, row)
        used = 0
        if method == "quadrature":
            if key not in orbits:
                offset = np.sqrt(key)*np.tan(np.deg2rad(full.fov_degrees)/2)/full.width
                canonical = full.momentum(metric, forward+offset*right)
                try:
                    orbits[key] = checked_quadrature(LayerOrbit, metric, camera.event, canonical, settings)
                except (ValueError, FloatingPointError, IntegrationWarning):
                    orbits[key] = None
            trajectory = orbits[key]
            if trajectory is not None:
                transverse = None if not key else (ix*right+iy*up)/np.sqrt(key)
                try:
                    hits, phases, truncated = checked_quadrature(trajectory.crossings, scene.surface, transverse, maximum)
                    records = pack_crossings(metric, p, scene.motion_observer, scene, hits, phases=phases)
                    records["geometry_valid"] &= records["null_error"] <= settings.null_tolerance
                    records["emission_valid"] &= records["geometry_valid"]
                    bad = ~records["geometry_valid"]
                    records["g"][bad], records["bolometric_intensity"][bad], records["observed_temperature"][bad] = np.nan, 0., np.nan
                    records["emitting_ordinal"] = np.where(records["emission_valid"], np.cumsum(records["emission_valid"]), 0)
                    status = 1 if trajectory.outcome == "escaped" else 0
                    reason = status
                    if trajectory.end_affine > settings.max_affine:
                        status, reason = 2, 2
                    direction = np.full(3, np.nan) if status != 1 else np.cos(trajectory.angle.background)*trajectory.radial+np.sin(trajectory.angle.background)*transverse
                    if transverse is None:
                        direction = np.full(3, np.nan)  # inward radial capture
                    if bad.any():
                        status, reason, direction = 2, 3, np.full(3, np.nan)
                    error = float(records["null_error"].max()) if len(records) else 0.
                    used = 1
                except CoplanarRay:
                    records, status, reason, error, direction, truncated = np.empty(0, CROSSING_DTYPE), 2, 4, np.nan, np.full(3, np.nan), False
                    used = 1
                except (ValueError, FloatingPointError, IntegrationWarning):
                    used = 0
        if not used:
            sol = cache.solution(global_row, global_col, p)
            records, status, reason, error, direction, truncated = collect_solution(metric, camera.event, p, settings, sol, scene, maximum)
        values.append((int(index), records, status, reason, error, direction, truncated, used))
    return values, (cache.integrated_rays+len(orbits), cache.steps, cache.evaluations)
