"""Systematic Schwarzschild lensing audit and matched diagnostic renders."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import tomllib

import numpy as np
from scipy.optimize import brentq

from .background import make_background
from .camera import Camera
from .cli import load_configuration
from .diagnostic_render import render_pair, shade, save_image
from .geodesic import Integration, trace, asymptotic_direction
from .metric import Schwarzschild
from .schwarzschild_orbit import orbit, ray_map


def angular_error(a, b):
    return float(np.arctan2(np.linalg.norm(np.cross(a, b)), np.dot(a, b)))


def compare_ray(metric, camera, settings, angle):
    """Compare final direction AND unwrapped azimuth, including the infinity tail."""
    forward, right, up = camera.axes()
    radial = -forward
    transverse = (right+2*up)/np.sqrt(5)
    p = camera.momentum(metric, np.cos(angle)*forward+np.sin(angle)*transverse)
    result, sol = trace(metric, camera.event, p, settings, return_solution=True)
    expected = "captured" if angle < metric.shadow_angle(np.linalg.norm(camera.position)) else "escaped"
    report = dict(status=result.status, expected=expected, reason=result.reason,
                  raw_status=result.raw_status, critical_uncertainty=result.critical_uncertainty,
                  null_error=result.max_null_error, energy_drift=result.energy_drift,
                  angular_momentum_drift=result.angular_momentum_drift)
    if expected == "escaped":
        reference = orbit(angle, np.linalg.norm(camera.position)/metric.mass, epsabs=1e-12, epsrel=1e-12)
        report.update(quadrature_phase=reference.phase, phase_turns=reference.phase/(2*np.pi),
                      quadrature_turning_radius=reference.turning_radius*metric.mass,
                      quadrature_error_estimate=reference.error_estimate)
        if result.raw_status == "escaped":
            # Retain raw errors for the audit even when the reliability guard
            # rejects that ray. Such a direction is never used by the renderer.
            raw_direction = asymptotic_direction(metric, sol.y[:, -1])
            direction = reference.direction[0]*transverse+reference.direction[2]*radial
            report["direction_error_radians"] = angular_error(direction, raw_direction)
            times = np.sort(np.r_[sol.t, (sol.t[1:]+sol.t[:-1])/2])
            states = sol.sol(times).T
            phases = np.unwrap(np.arctan2(states[:, 1:4] @ transverse, states[:, 1:4] @ radial))
            end_radial = states[-1, 1:4]/np.linalg.norm(states[-1, 1:4])
            end_tangent = -np.sin(phases[-1])*radial+np.cos(phases[-1])*transverse
            tail = np.arctan2(np.dot(raw_direction, end_tangent), np.dot(raw_direction, end_radial))
            report["ode_phase"] = float(phases[-1]+tail)
            report["phase_error_radians"] = float(abs(report["ode_phase"]-reference.phase))
            report["max_unwrap_step_radians"] = float(np.max(np.abs(np.diff(phases))))
            def radial_speed(t):
                state = sol.sol(t)
                return np.dot(state[1:4], (metric.inverse(state[:4]) @ state[4:])[1:])
            speeds = np.array([radial_speed(t) for t in sol.t])
            brackets = np.flatnonzero((speeds[:-1] < 0) & (speeds[1:] >= 0))
            if len(brackets):
                i = brackets[0]
                turning_time = brentq(radial_speed, sol.t[i], sol.t[i+1], xtol=1e-10)
                report["ode_turning_radius"] = float(np.linalg.norm(sol.sol(turning_time)[1:4]))
                report["turning_radius_error"] = abs(report["ode_turning_radius"]-report["quadrature_turning_radius"])
    return report


def numerical_shadow(radius, settings):
    metric = Schwarzschild()
    camera = Camera(position=(0, 0, radius))
    critical = metric.shadow_angle(radius)
    low, high = critical*0.98, critical*1.02
    settings = replace(settings, rtol=1e-11, atol=1e-13,
                       escape_radius=max(2*radius, 100), max_affine=5*radius+500)
    for _ in range(20):
        angle = (low+high)/2
        result = trace(metric, camera.event, camera.momentum(metric, [np.sin(angle), 0, -np.cos(angle)]), settings,
                       critical_guard=False)  # explicitly measure the raw ODE separatrix
        if result.status == "unresolved":
            raise RuntimeError(f"unresolved shadow bisection: {result.reason}")
        if result.status == "captured":
            low = angle
        else:
            high = angle
    return low, high


def systematic_checks(metric, camera, settings):
    radius = np.linalg.norm(camera.position)
    critical = metric.shadow_angle(radius)
    maximum = np.arctan(np.tan(np.deg2rad(camera.fov_degrees)/2)*np.hypot(1, camera.height/camera.width))
    ring_angle = brentq(lambda a: orbit(a, radius/metric.mass).phase-np.pi,
                        critical*1.01, maximum, xtol=1e-14)
    cases = [("capture", critical*0.8), ("near_capture", critical*(1-1e-8))]
    for epsilon in (1e-2, 1e-4, 1e-6, 1e-8, 1e-10):
        angle = np.arcsin((1+epsilon)*np.sin(critical))
        cases.append((f"critical_b_plus_{epsilon:g}", angle))
    cases += [("inside_outer_arc", ring_angle*0.8), ("outer_alignment_ring", ring_angle),
              ("outside_outer_arc", ring_angle*1.2), ("frame_edge", maximum),
              ("weak_control_outside_frame", 0.8)]
    cross_checks = []
    for label, angle in cases:
        rows = []
        for tol in (1e-8, 1e-10, 1e-12):
            row = compare_ray(metric, camera, replace(settings, rtol=tol, atol=tol/100), angle)
            rows.append(dict(rtol=tol, atol=tol/100, **row))
        item = dict(case=label, angle_radians=float(angle), runs=rows)
        if angle > critical:
            phases = [orbit(angle, radius/metric.mass, epsabs=eps, epsrel=eps).phase for eps in (1e-8, 1e-10, 1e-12)]
            perturb = np.spacing(angle)
            sensitivity = abs(orbit(angle+perturb, radius/metric.mass).phase-orbit(angle-perturb, radius/metric.mass).phase)/2
            item.update(quadrature_tolerances=[1e-8, 1e-10, 1e-12], quadrature_phases=phases,
                        one_ulp_angle_phase_sensitivity=float(sensitivity))
        cross_checks.append(item)
    scale = camera.width/(2*np.tan(np.deg2rad(camera.fov_degrees)/2))
    rings = []
    for order in range(1, 7):
        target = order*np.pi
        angle = brentq(lambda a: orbit(a, radius/metric.mass).phase-target,
                       critical*(1+1e-12), maximum, xtol=5e-16, rtol=1e-14)
        pixel = scale*np.tan(angle)
        rings.append(dict(phase_pi=order, angle_degrees=float(np.rad2deg(angle)),
                          radius_pixels=float(pixel), separation_from_shadow_pixels=float(pixel-scale*np.tan(critical))))
    shadows = []
    for distance in (20.0, 100.0, 400.0):
        low, high = numerical_shadow(distance, settings)
        analytic = Schwarzschild().shadow_angle(distance)
        for fov in (32.0, 60.0):
            factor = camera.width/(2*np.tan(np.deg2rad(fov)/2))
            shadows.append(dict(radius_M=distance, fov_degrees=fov,
                                expected_angle_degrees=float(np.rad2deg(analytic)),
                                ode_angle_bracket_degrees=np.rad2deg([low, high]).tolist(),
                                expected_radius_pixels=float(factor*np.tan(analytic)),
                                ode_radius_bracket_pixels=(factor*np.tan([low, high])).tolist()))
    return dict(cross_checks=cross_checks, alignment_rings=rings, shadow_sweep=shadows,
                expected_shadow_angle_degrees=float(np.rad2deg(critical)),
                expected_shadow_radius_pixels=float(scale*np.tan(critical)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("examples/composition.toml"))
    parser.add_argument("--output", type=Path, default=Path("examples/validated"))
    parser.add_argument("--samples", type=int, default=4)
    args = parser.parse_args()
    metric, camera, settings = load_configuration(args.config)
    config = tomllib.loads(args.config.read_text(encoding="utf-8"))
    report = systematic_checks(metric, camera, settings)
    print("Trajectory and shadow checks completed", flush=True)
    directions, status, distinct, images = render_pair(metric, camera, args.output, config["background"], args.samples)
    print("Matched renders completed", flush=True)
    window = (camera.height//2-24, camera.height//2+24, camera.width//2-24, camera.width//2+24)
    inner_directions, inner_status, _ = ray_map(metric, camera, samples=16, window=window)
    grid = make_background(dict(config["background"], mode="grid"))
    save_image(args.output/"inner_grid.png", shade(inner_directions, inner_status, grid))
    report["inner_grid"] = dict(window_pixels=window, pixels_per_original_pixel=16,
                                note="Fresh rays in the same camera window, not an enlarged low-resolution bitmap")
    del inner_directions, inner_status, grid
    report.update(configuration=config, samples_per_axis=args.samples, distinct_orbits=distinct,
                  subpixel_counts={k: int(np.count_nonzero(status == v)) for v, k in enumerate(("captured", "escaped", "unresolved"))})
    h, w = status.shape
    yy, xx = np.mgrid[:h, :w]
    pixels = np.hypot(xx+0.5-w/2, yy+0.5-h/2)/args.samples
    report["rendered_shadow_radius_bracket_pixels"] = [float(pixels[status == 0].max()), float(pixels[status == 1].min())]
    report["shadow_area_equivalent_radius_pixels"] = float(np.sqrt(np.count_nonzero(status == 0)/np.pi)/args.samples)
    del directions, status, pixels, yy, xx
    report["sampling"] = []
    yy, xx = np.mgrid[:camera.height, :camera.width]
    radius_pixels = np.hypot(xx+0.5-camera.width/2, yy+0.5-camera.height/2)
    regions = {"all": np.ones(radius_pixels.shape, dtype=bool),
               "near_shadow": (radius_pixels > report["expected_shadow_radius_pixels"]) & (radius_pixels < 25),
               "outer_arc": (radius_pixels > 90) & (radius_pixels < 140)}
    for samples in (1, 2):
        dirs, codes, _ = ray_map(metric, camera, samples)
        for mode in ("starfield", "grid"):
            background = make_background(dict(config["background"], mode=mode))
            rgb = shade(dirs, codes, background, samples)
            error = np.abs(rgb-images[mode]).mean(axis=-1)
            report["sampling"].append(dict(mode=mode, samples_per_axis=samples,
                                           reference_samples_per_axis=args.samples,
                                           display_rgb_mae={name: float(error[mask].mean()) for name, mask in regions.items()}))
            save_image(args.output/f"{mode}_{samples}x.png", rgb)
            del background
        del dirs, codes
    (args.output/"validation.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(f"Saved matched images and validation.json in {args.output}")


if __name__ == "__main__":
    main()
