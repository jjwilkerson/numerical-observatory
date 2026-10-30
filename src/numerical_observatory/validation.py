"""Reproducible numerical report: python -m numerical_observatory.validation."""
import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from .camera import Camera
from .geodesic import Integration, trace
from .metric import Schwarzschild
from .schwarzschild_orbit import orbit


def orbit_direction(angle, radius=20.0):
    """Independent Schwarzschild orbit quadrature for an inward escaping ray.

    Uses the outer turning point and conserved impact parameter, without the
    Hamiltonian ODE. Restricted to M=1 and a camera on the positive z axis.
    """
    return orbit(angle, radius).direction


def validation_report():
    metric = Schwarzschild()
    settings = Integration()
    def ray(angle, radius=20.0, config=settings, critical_guard=True):
        camera = Camera(position=(0, 0, radius))
        return trace(metric, camera.event,
                     camera.momentum(metric, [np.sin(angle), 0, -np.cos(angle)]), config,
                     critical_guard=critical_guard)
    boundaries = []
    for radius in (8.0, 20.0, 50.0):
        analytic = metric.shadow_angle(radius)
        low, high = 0.95*analytic, 1.05*analytic
        for _ in range(20):
            mid = (low+high)/2
            result = ray(mid, radius, critical_guard=False)
            if result.status == "unresolved":
                raise RuntimeError(f"Unresolved boundary probe: {result.reason}")
            if result.status == "captured":
                low = mid
            else:
                high = mid
        measured = radius*np.sin((low+high)/2)/np.sqrt(1-2/radius)
        boundaries.append(dict(camera_radius=radius, analytic_angle_radians=float(analytic),
                               numerical_angle_bracket=[float(low), float(high)],
                               measured_impact_parameter=float(measured),
                               relative_impact_error=float(abs(measured/metric.critical_impact_parameter-1))))
    angle = metric.shadow_angle(20)*1.015
    reference = orbit_direction(angle)
    convergence = []
    for tol in (1e-6, 1e-9, 1e-12):
        result = ray(angle, config=replace(settings, rtol=tol, atol=tol/100))
        if result.status != "escaped":
            raise RuntimeError(f"Convergence probe: {result.reason}")
        convergence.append(dict(rtol=tol, atol=tol/100,
                                direction_error_vs_orbit_quadrature=float(np.linalg.norm(result.direction-reference)),
                                max_null_error=result.max_null_error,
                                angular_momentum_drift=result.angular_momentum_drift,
                                energy_drift=result.energy_drift))
    return dict(critical_impact_parameter=float(metric.critical_impact_parameter),
                boundary_validation=boundaries, convergence=convergence,
                reference="independent Schwarzschild turning-point orbit quadrature",
                convergence_ray_angle_radians=float(angle))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("output/validation.json"))
    args = parser.parse_args()
    report = validation_report()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
