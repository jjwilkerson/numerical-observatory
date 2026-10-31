"""Explicit small ODE audits of the angle diagnostic; not an image renderer."""
from dataclasses import replace
import time

import numpy as np
from scipy.integrate import quad

from .geodesic import Integration, trace, asymptotic_direction
from .metric import Schwarzschild
from .orbital_angle import OrbitalAngle, orbital_frame
from .surfaces import ThinAnnulus


def dense_angle(metric, solution, event, momentum, stop=None):
    """Independent geometry unwrap AND integral of actual ODE angular speed.

    Uses the existing Hamiltonian dense trajectory, never radial quadrature or
    a separate geodesic ODE. Midpoints diagnose aliasing; the velocity integral
    independently checks that unwrap has not dropped full turns.
    """
    _, _, _, radial, transverse = orbital_frame(metric, event, momentum)
    stop = float(solution.t[-1] if stop is None else stop)
    ends = np.r_[solution.t[solution.t < stop], stop]
    times = np.sort(np.r_[ends, (ends[1:]+ends[:-1])/2])
    states = solution.sol(times).T
    if transverse is None:
        phases = np.zeros(len(times))
    else:
        phases = np.unwrap(np.arctan2(states[:, 1:4] @ transverse, states[:, 1:4] @ radial))
        phases -= phases[0]
    def speed(t):
        y = solution.sol(t)
        x, p = y[1:4], y[4:]
        v = (metric.inverse(y[:4]) @ p)[1:]
        return np.linalg.norm(np.cross(x, v))/np.dot(x, x)
    integral = sum(quad(speed, low, high, epsabs=1e-11, epsrel=1e-11)[0] for low, high in zip(ends[:-1], ends[1:]))
    maximum = float(np.max(np.abs(np.diff(phases)))) if len(phases) > 1 else 0.
    if maximum >= np.pi/2 or abs(integral-phases[-1]) > 1e-7:
        raise ValueError("dense orbital-angle unwrap/rate check failed")
    return dict(angle=float(phases[-1]), rate_integral=float(integral), max_sample_increment=maximum,
                unwrap_rate_difference=float(abs(integral-phases[-1])))


def audit_ray(metric, event, momentum, settings, surface=None):
    geometry = OrbitalAngle.from_initial(metric, event, momentum, eps=1e-12)
    radius, impact, _, radial, transverse = orbital_frame(metric, event, momentum)
    result, solution = trace(metric, event, momentum, settings, return_solution=True)
    record = dict(ode_status=result.status, ode_reason=result.reason, null_error=result.max_null_error,
                  impact_parameter_M=impact/metric.mass, quadrature_outcome=geometry.outcome)
    if surface is not None:
        hit = surface.first_hit(solution)
        if hit is not None:
            emit = dense_angle(metric, solution, event, momentum, hit.affine)
            phi, branch, error = geometry.saved_hit(hit.event[1:], radial, transverse)
            record.update(crossing=hit.crossing_index, hit_position=hit.event[1:].tolist(),
                          phi_emit_ode=emit["angle"], phi_emit_quadrature=phi,
                          phi_emit_error_radians=abs(phi-emit["angle"]), emission_leg=branch,
                          emission_dense_checks=emit, hit_geometry_error_radians=error)
    if result.status == "escaped":
        finite = dense_angle(metric, solution, event, momentum)
        y = solution.y[:, -1]
        n = y[1:4]/np.linalg.norm(y[1:4])
        direction = asymptotic_direction(metric, y)
        if transverse is None:
            tail = 0.
        else:
            tangent = np.cross(np.cross(radial, transverse), n)
            tail = float(np.arctan2(np.dot(direction, tangent), np.dot(direction, n)))
        phi = finite["angle"]+tail
        record.update(phi_background_ode=phi, phi_background_quadrature=geometry.background,
                      phi_background_error_radians=abs(phi-geometry.background),
                      finite_escape_angle=finite["angle"], infinity_tail_radians=tail, background_dense_checks=finite)
    return record


def pixel_momentum(camera, metric, x, y):
    """Same pinhole arithmetic as Camera.rays, including diagnostic windows."""
    w, h = getattr(camera, "reference_width", camera.width), getattr(camera, "reference_height", camera.height)
    x, y = x+getattr(camera, "origin_x", 0), y+getattr(camera, "origin_y", 0)
    forward, right, up = camera.axes()
    half = np.tan(np.deg2rad(camera.fov_degrees)/2)
    horizontal = (2*(x+.5)/w-1)*half
    vertical = (1-2*(y+.5)/h)*half*h/w
    return camera.momentum(metric, forward+horizontal*right+vertical*up)


def validate_render(metadata, raw, angle_arrays=None):
    started = time.perf_counter()
    from .orbital_diagnostics import saved_camera, sampled_camera
    camera = sampled_camera(saved_camera(metadata), int(raw.get("samples", 1)))
    metric = Schwarzschild(metadata["mass"])
    surface = ThinAnnulus(**metadata["geometry"])
    original = Integration(**metadata["integration"])
    points = [(310, 192), (313, 192), (266, 240), (269, 240), (373, 240), (286, 205), (287, 205),
              (284, 272), (285, 272), (311, 287), (200, 245), (319, 245), (319, 220), (0, 0), (320, 321)]
    points = [(x, y) for x, y in points if x < camera.width and y < camera.height]
    cases = []
    for x, y in points:
        p = pixel_momentum(camera, metric, x, y)
        runs = []
        for tolerance in (original.rtol, min(original.rtol, 1e-11)):
            settings = replace(original, rtol=tolerance, atol=min(original.atol, tolerance/100))
            audit = audit_ray(metric, camera.event, p, settings, surface)
            audit["rtol"] = tolerance
            if "phi_emit_ode" in audit and raw["status"][y, x] == 3:
                audit["saved_vs_ode_hit_position_difference_M"] = float(np.linalg.norm(np.array(audit["hit_position"])-raw["hit_position"][y, x])/metric.mass)
                if int(raw["crossing_index"][y, x]) != audit["crossing"]:
                    raise AssertionError("independent ODE first emitting crossing disagrees with saved data")
            if angle_arrays is not None:
                if "phi_emit_ode" in audit:
                    audit["saved_map_vs_ode_emit_error_radians"] = float(abs(angle_arrays["phi_emit_radians"][y, x]-audit["phi_emit_ode"]))
                if "phi_background_ode" in audit:
                    audit["saved_map_vs_ode_background_error_radians"] = float(abs(angle_arrays["phi_background_radians"][y, x]-audit["phi_background_ode"]))
            runs.append(audit)
        cases.append(dict(x=x, y=y, saved_status=int(raw["status"][y, x]), runs=runs))
    tight = [case["runs"][-1] for case in cases]
    maxima = {field: max((row[field] for row in tight if field in row), default=0.) for field in (
        "phi_emit_error_radians", "phi_background_error_radians", "saved_map_vs_ode_emit_error_radians",
        "saved_map_vs_ode_background_error_radians", "saved_vs_ode_hit_position_difference_M")}
    if any(value > 1e-6 for field, value in maxima.items() if "radians" in field):
        raise AssertionError(f"ODE angle audit exceeded 1e-6 radians: {maxima}")
    return dict(cases=cases, tight_run_maxima=maxima, ode_integrations=2*len(cases), elapsed_seconds=time.perf_counter()-started,
                scope="explicit small transparent ODE audit; physical emission phase stops at first valid opaque hit",
                validation_tolerance_radians=1e-6, image_rerendered=False)
