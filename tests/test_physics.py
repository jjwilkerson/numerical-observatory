from dataclasses import replace

import numpy as np
import pytest

from numerical_observatory.camera import Camera
from numerical_observatory.geodesic import Integration, hamilton_rhs, trace
from numerical_observatory.metric import ETA, Schwarzschild


def angular_ray(angle, settings=Integration(), radius=20.0, mass=1.0):
    metric = Schwarzschild(mass)
    camera = Camera(position=(0, 0, radius))
    p = camera.momentum(metric, [np.sin(angle), 0, -np.cos(angle)])
    return trace(metric, camera.event, p, settings)


def test_inverse_derivatives_and_horizon_regularity():
    metric = Schwarzschild()
    for radius in (1.99, 2.0, 3.0, 20.0):
        event = np.array([7.0, 0.6*radius, 0, 0.8*radius])
        np.testing.assert_allclose(metric.covariant(event) @ metric.inverse(event), np.eye(4), atol=2e-15)
        derivative = metric.inverse_derivative(event)
        for axis in range(4):
            delta = np.eye(4)[axis] * 1e-5
            fd = (metric.inverse(event + delta) - metric.inverse(event - delta)) / 2e-5
            np.testing.assert_allclose(derivative[axis], fd, atol=1e-9)


def test_tetrad_and_initial_null_rays():
    metric = Schwarzschild()
    camera = Camera(position=(8, 4, 15), width=7, height=5)
    frame = camera.tetrad(metric)
    np.testing.assert_allclose(frame @ metric.covariant(camera.event) @ frame.T, ETA, atol=1e-14)
    for _, _, p in camera.rays(metric):
        assert abs(p @ metric.inverse(camera.event) @ p) < 1e-14
        assert p[0] > 0  # negative conserved energy = past-directed


@pytest.mark.parametrize("radius", [8.0, 20.0, 50.0])
def test_finite_distance_critical_boundary(radius):
    metric = Schwarzschild()
    critical = metric.shadow_angle(radius)
    assert angular_ray(critical * (1 - 1e-4), radius=radius).status == "captured"
    assert angular_ray(critical * (1 + 1e-4), radius=radius).status == "escaped"
    # Recover boundary by classification, independently of image pixel sampling.
    low, high = critical*0.95, critical*1.05
    for _ in range(18):
        mid = (low + high)/2
        result = angular_ray(mid, radius=radius)
        assert result.status != "unresolved"
        if result.status == "captured":
            low = mid
        else:
            high = mid
    measured_b = radius * np.sin((low+high)/2) / np.sqrt(1-2/radius)
    assert abs(measured_b / metric.critical_impact_parameter - 1) < 3e-7


def test_radial_capture_escape_and_budget():
    assert angular_ray(0).status == "captured"
    outward = angular_ray(np.pi)
    assert outward.status == "escaped"
    np.testing.assert_allclose(outward.direction, [0, 0, 1], atol=1e-12)
    limited = angular_ray(0, replace(Integration(), max_affine=0.1))
    assert limited.status == "unresolved" and limited.reason == "affine_limit"
    rejected = angular_ray(0.4, replace(Integration(), null_tolerance=1e-16))
    assert rejected.status == "unresolved" and rejected.reason == "null_constraint_exceeded"


def test_convergence_and_conserved_quantities():
    angle = Schwarzschild().shadow_angle(20) * 1.015
    results = [angular_ray(angle, replace(Integration(), rtol=tol, atol=tol/100))
               for tol in (1e-6, 1e-9, 1e-12)]
    assert all(r.status == "escaped" for r in results)
    coarse, fine, reference = results
    coarse_error = np.linalg.norm(coarse.direction-reference.direction)
    fine_error = np.linalg.norm(fine.direction-reference.direction)
    assert fine_error < coarse_error / 20
    assert fine_error < 1e-6
    assert fine.max_null_error < coarse.max_null_error / 20
    assert reference.max_null_error < 1e-9
    assert reference.energy_drift < 1e-14
    assert reference.angular_momentum_drift < 1e-10


def test_escape_sphere_independence():
    a = angular_ray(0.4, replace(Integration(), escape_radius=60, rtol=1e-11, atol=1e-13))
    b = angular_ray(0.4, replace(Integration(), escape_radius=160, rtol=1e-11, atol=1e-13))
    np.testing.assert_allclose(a.direction, b.direction, atol=1e-9)


def test_event_interface_uses_time_derivatives():
    class TimeMetric:
        def inverse(self, event):
            return np.diag([-1.0, 1+event[0], 1.0, 1.0])
        def inverse_derivative(self, event):
            result = np.zeros((4, 4, 4))
            result[0, 1, 1] = 1.0
            return result
    rhs = hamilton_rhs(TimeMetric(), np.array([0, 0, 0, 0, 1, 1, 0, 0]))
    assert rhs[4] == -0.5
