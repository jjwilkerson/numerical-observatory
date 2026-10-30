"""Independent escaping-orbit disk intersection reference and ODE validation.

This partial-orbit quadrature is an audit tool, not the disk renderer. Production
uses the general ODE, including captured rays, noncentral and moving cameras.
"""
from dataclasses import replace
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq

from .camera import Camera
from .disk import make_disk
from .emission import frequency_shift
from .geodesic import Integration, trace
from .metric import Schwarzschild
from .schwarzschild_orbit import orbit
from .surfaces import SurfaceHit


def quadrature_hit(metric, event, momentum, surface):
    """First annulus hit for an inward, ESCAPING Schwarzschild null geodesic.

    Reconstruct the full event and covector from E,L, radial branch and orbital
    quadratures. Compare against ODE; do not apply this to captured trajectories.
    """
    if type(metric) is not Schwarzschild:
        raise ValueError("Schwarzschild-only quadrature reference")
    r_obs = np.linalg.norm(event[1:4])
    radial = event[1:4]/r_obs
    angular = np.cross(event[1:4], momentum[1:4])
    length = np.linalg.norm(angular)
    energy = momentum[0]  # backward-ray Killing covector component is positive
    b = length/energy
    velocity = metric.inverse(event) @ momentum
    if (energy <= 0 or b <= metric.critical_impact_parameter
            or np.dot(velocity[1:], radial) >= 0 or r_obs <= surface.outer_radius):
        raise ValueError("reference requires inward escaping rays from outside the annulus")
    transverse = np.cross(angular/length, radial)
    angle = np.arcsin(b*np.sqrt(1-2*metric.mass/r_obs)/r_obs)
    solution = orbit(angle, r_obs/metric.mass, epsabs=1e-12, epsrel=1e-12)
    turn = solution.turning_radius*metric.mass
    if turn > surface.outer_radius:
        return None
    ut = 1/solution.turning_radius
    delta = (solution.turning_radius-3)/solution.turning_radius
    def phase_integrand(s):
        u = ut*(1-s*s)
        return 2*np.sqrt(ut)/np.sqrt((ut+u)*delta+(ut-u)*(ut+2*u))
    def phase(s):
        return quad(phase_integrand, 0, s, epsabs=1e-12, epsrel=1e-12)[0]
    s_obs = np.sqrt(1-turn/r_obs)
    phase_in = phase(s_obs)
    s_outer = np.sqrt(1-turn/surface.outer_radius)
    s_inner = np.sqrt(max(0, 1-turn/surface.inner_radius))
    outer_phase, inner_phase = phase(s_outer), phase(s_inner)
    a, b_plane = np.dot(surface.normal, radial), np.dot(surface.normal, transverse)
    if np.hypot(a, b_plane) < 1e-12:
        raise ValueError("coplanar reference ray")
    first = np.arctan2(-a, b_plane) % np.pi
    if first < 1e-12:
        first += np.pi
    count = int(np.floor((solution.phase-first)/np.pi))+1
    for index in range(max(0, count)):
        phi = first+index*np.pi
        target = abs(phi-phase_in)
        if target < inner_phase-1e-12 or target > outer_phase+1e-12:
            continue
        s = brentq(lambda s: phase(s)-target, 0, s_outer, xtol=1e-13)
        radius = turn/(1-s*s)
        n = np.cos(phi)*radial+np.sin(phi)*transverse
        tangent = -np.sin(phi)*radial+np.cos(phi)*transverse
        branch = -1 if phi < phase_in else 1
        def components(s, sign):
            r = turn/(1-s*s)
            f, lapse = 2*metric.mass/r, 1-2*metric.mass/r
            dr = sign*energy*np.sqrt(max(0, 1-lapse*(length/energy/r)**2))
            pr = (dr+f*energy)/lapse
            dt = -(1+f)*energy-f*pr
            return r, pr, dt
        _, pr, _ = components(s, branch)
        covector = np.r_[energy, pr*n+(length/radius)*tangent]
        def elapsed(lo, hi, sign, coordinate_time):
            def integrand(s):
                r, _, dt = components(s, sign)
                return (dt if coordinate_time else 1)*r*r/length*phase_integrand(s)
            return quad(integrand, lo, hi, epsabs=1e-9, epsrel=1e-11)[0]
        if branch < 0:
            time = event[0]+elapsed(s, s_obs, -1, True)
            affine = elapsed(s, s_obs, -1, False)
        else:
            time = event[0]+elapsed(0, s_obs, -1, True)+elapsed(0, s, 1, True)
            affine = elapsed(0, s_obs, -1, False)+elapsed(0, s, 1, False)
        return SurfaceHit(affine, np.r_[time, radius*n], covector, index+1)
    return None


def validation_report():
    metric = Schwarzschild()
    camera = Camera(position=(0, 0, 60), width=40, height=30)
    settings = Integration(rtol=1e-9, atol=1e-11, escape_radius=110, max_affine=400, max_step=4)
    report = {"cases": [], "scope": "quadrature reference restricted to inward escaping rays; production disk renderer uses ODE"}
    observer = camera.tetrad(metric)[0]
    # Deterministically find direct and secondary examples, then tighten ODE
    # tolerances for a compact sample of hits and geometrical non-hits.
    for inclination in (60., 85.):
        scene = make_disk(metric, dict(enabled=True, inclination_degrees=inclination))
        selected, orders = [], set()
        for row, col, p in camera.rays(metric):
            impact = np.linalg.norm(np.cross(camera.event[1:], p[1:]))/p[0]
            if impact <= metric.critical_impact_parameter:
                continue
            ref = quadrature_hit(metric, camera.event, p, scene.surface)
            order = 0 if ref is None else ref.crossing_index
            if order not in orders or (order == 1 and len(selected) < 5 and col % 9 == 0):
                orders.add(order)
                selected.append((row, col, p, ref))
        # Coarse image pixels need not sample very narrow higher-order images.
        for epsilon in np.geomspace(2e-5, 0.01, 40):
            angle = np.arcsin(np.sin(metric.shadow_angle(60))*(1+epsilon))
            p = camera.momentum(metric, [0, np.sin(angle), -np.cos(angle)])
            ref = quadrature_hit(metric, camera.event, p, scene.surface)
            if ref is not None and ref.crossing_index >= 3:
                selected.append((-1, -1, p, ref))
                break
        for row, col, p, ref in selected:
            entry = dict(inclination=inclination, row=row, col=col,
                         initial_event=camera.event.tolist(), initial_momentum=p.tolist(),
                         reference_crossing=None if ref is None else ref.crossing_index, runs=[])
            for tol in (1e-7, 1e-9, 1e-11):
                result = trace(metric, camera.event, p, replace(settings, rtol=tol, atol=tol/100), surface=scene.surface)
                run = dict(rtol=tol, status=result.status, null_error=result.max_null_error)
                if ref is None:
                    if result.status == "emitted":
                        raise AssertionError("ODE-only disk hit")
                else:
                    if result.status != "emitted" or result.hit.crossing_index != ref.crossing_index:
                        raise AssertionError("first-hit disagreement")
                    g_ref = frequency_shift(p, observer, ref.momentum, scene.motion.four_velocity(ref.event))
                    g_ode = frequency_shift(p, observer, result.hit.momentum, scene.motion.four_velocity(result.hit.event))
                    run.update(position_error=float(np.linalg.norm(result.hit.event[1:]-ref.event[1:])),
                               time_error=float(abs(result.hit.event[0]-ref.event[0])),
                               affine_error=float(abs(result.hit.affine-ref.affine)),
                               momentum_error=float(np.linalg.norm(result.hit.momentum-ref.momentum)),
                               g_error=float(abs(g_ref-g_ode)), g=g_ode,
                               crossing_index=result.hit.crossing_index)
                entry["runs"].append(run)
            report["cases"].append(entry)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("examples/disks/validation.json"))
    args = parser.parse_args()
    report = validation_report()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
