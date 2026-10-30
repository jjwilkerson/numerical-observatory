from dataclasses import replace

import numpy as np
import pytest

from numerical_observatory.background import CelestialGrid, make_background
from numerical_observatory.camera import Camera
from numerical_observatory.geodesic import Integration, TraceDomain, trace
from numerical_observatory.lensing_validation import compare_ray, angular_error
from numerical_observatory.metric import ETA, Schwarzschild
from numerical_observatory.render import render
from numerical_observatory.schwarzschild_orbit import compatible, orbit, ray_map
from numerical_observatory.symmetry import CenteredRayCache


class FlatAnisotropic:
    # mass deliberately present: optimizations must not duck-type their way in.
    mass = 1.0
    def covariant(self, event):
        return np.diag([-1., 2., 3., 4.])
    def inverse(self, event):
        return np.diag([-1., 1/2, 1/3, 1/4])
    def inverse_derivative(self, event):
        return np.zeros((4, 4, 4))


def flat_domain(valid_region=None):
    return TraceDomain(lambda x: 1.0, lambda x: x[3]-2,
                       lambda state: np.sqrt([2, 3, 4])*(FlatAnisotropic().inverse(state[:4]) @ state[4:])[1:],
                       valid_region)


def test_general_camera_and_ode_fallback(tmp_path):
    metric = FlatAnisotropic()
    camera = Camera(position=(0, 0, 0), target=(0, 0, 1), width=3, height=3)
    frame = camera.tetrad(metric)
    np.testing.assert_allclose(frame @ metric.covariant(camera.event) @ frame.T, ETA, atol=1e-14)
    assert not compatible(metric, camera)
    assert not CenteredRayCache(metric, camera, Integration()).enabled
    with pytest.raises(ValueError, match="incompatible"):
        ray_map(metric, camera)
    with pytest.raises(ValueError, match="TraceDomain"):
        trace(metric, camera.event, camera.momentum(metric, [0, 0, 1]), Integration())
    report = render(metric, camera, Integration(), tmp_path, domain=flat_domain(),
                    method="quadrature", background=CelestialGrid())
    assert report["method"] == "ode" and report["requested_method"] == "quadrature"
    assert report["counts"] == dict(captured=0, escaped=9, unresolved=0)
    assert report["maxima"]["energy_drift"] is None
    with np.load(tmp_path/"rays.npz") as rays:
        np.testing.assert_allclose(rays["directions"][1, 1], [0, 0, 1], atol=1e-14)


def test_data_domain_exit_is_unresolved():
    metric = FlatAnisotropic()
    camera = Camera(position=(0, 0, 0), target=(0, 0, 1))
    result = trace(metric, camera.event, camera.momentum(metric, [0, 0, 1]), Integration(),
                   domain=flat_domain(lambda x: x[0]+0.5))
    assert result.status == "unresolved" and result.reason == "data_domain_exit"


def test_general_ode_fallback_supports_subpixel_sampling(tmp_path):
    camera = Camera(position=(0, 0, 0), target=(0, 0, 1), width=1, height=1)
    report = render(FlatAnisotropic(), camera, Integration(), tmp_path,
                    domain=flat_domain(), background=CelestialGrid(), method="quadrature", samples=2)
    assert report["method"] == "ode" and report["counts"]["escaped"] == 4
    assert report["samples_per_axis"] == 2


def test_time_dependent_metric_changes_energy_without_false_drift_alarm():
    class Expanding:
        def covariant(self, x):
            return np.diag([-1., *([np.exp(0.04*x[0])]*3)])
        def inverse(self, x):
            return np.diag([-1., *([np.exp(-0.04*x[0])]*3)])
        def inverse_derivative(self, x):
            result = np.zeros((4, 4, 4))
            result[0, 1:, 1:] = -0.04*np.exp(-0.04*x[0])*np.eye(3)
            return result
    metric = Expanding()
    camera = Camera(position=(0, 0, 0), target=(0, 0, 1))
    domain = TraceDomain(lambda x: 1., lambda x: x[3]-2,
                         lambda y: (metric.inverse(y[:4]) @ y[4:])[1:])
    p = camera.momentum(metric, [0, 0, 1])
    result, sol = trace(metric, camera.event, p, Integration(rtol=1e-11, atol=1e-13),
                        domain=domain, return_solution=True)
    assert result.status == "escaped" and result.max_null_error < 1e-8
    assert abs(sol.y[4, -1]-p[0]) > 0.01
    assert result.energy_drift is None and result.angular_momentum_drift is None


def test_optimization_guards_for_moving_and_off_center_observers():
    metric = Schwarzschild()
    camera = Camera()
    frame = camera.tetrad(metric)
    moving = replace(camera, observer_frame=tuple(map(tuple, frame)))
    assert not compatible(metric, moving)
    assert not CenteredRayCache(metric, moving, Integration()).enabled
    assert not compatible(metric, replace(camera, target=(1, 0, 0)))
    class AlteredSchwarzschild(Schwarzschild):
        pass
    assert not compatible(AlteredSchwarzschild(), camera)
    assert not CenteredRayCache(AlteredSchwarzschild(), camera, Integration()).enabled


def test_grid_mapping_is_spherical_periodic_and_camera_independent():
    grid = CelestialGrid(longitude_degrees=0, latitude_degrees=0, roll_degrees=0)
    directions = np.array([[1e-12, 0, 1], [-1e-12, 0, 1], [0, 1, 0], [0, -1, 0]])
    rgb = grid.color(directions)
    np.testing.assert_allclose(rgb[0], rgb[1], atol=1e-8)
    assert np.isfinite(rgb).all()
    assert isinstance(make_background(dict(mode="grid", exposure=3, resolution=8)), CelestialGrid)
    assert angular_error([1, 0, 0], [np.cos(2*np.pi), np.sin(2*np.pi), 0]) < 1e-14


def test_near_critical_multiple_winding_convergence():
    metric = Schwarzschild()
    camera = Camera(position=(0, 0, 400))
    angle = np.arcsin(np.sin(metric.shadow_angle(400))*(1+1e-6))
    settings = Integration(escape_radius=800, max_affine=2000, max_step=80)
    coarse = compare_ray(metric, camera, replace(settings, rtol=1e-8, atol=1e-10), angle)
    fine = compare_ray(metric, camera, replace(settings, rtol=1e-12, atol=1e-14), angle)
    assert fine["phase_turns"] > 2
    assert fine["status"] == "escaped" and fine["phase_error_radians"] < 2e-4
    assert fine["direction_error_radians"] < coarse["direction_error_radians"]/20
    assert fine["max_unwrap_step_radians"] < np.pi/2
    assert fine["turning_radius_error"] < 1e-6


def test_coarse_false_capture_is_flagged_unresolved():
    metric = Schwarzschild()
    camera = Camera(position=(0, 0, 400))
    angle = np.arcsin(np.sin(metric.shadow_angle(400))*(1+1e-10))
    settings = Integration(escape_radius=800, max_affine=2000, max_step=80)
    report = compare_ray(metric, camera, settings, angle)
    assert report["status"] == "unresolved"
    assert report["reason"] == "near_critical_uncertainty"
    fine = compare_ray(metric, camera, replace(settings, rtol=1e-12, atol=1e-14), angle)
    assert fine["status"] == "escaped"


def test_rendered_shadow_and_finite_distance_scaling():
    metric = Schwarzschild()
    camera = Camera(position=(0, 0, 400), fov_degrees=32, width=32, height=24)
    directions, status, _ = ray_map(metric, camera, samples=2)
    h, w = status.shape
    yy, xx = np.mgrid[:h, :w]
    angle = np.arctan(np.hypot(2*xx+1-w, h-2*yy-1)*np.tan(np.deg2rad(16))/w)
    np.testing.assert_array_equal(status == 0, angle < metric.shadow_angle(400))
    assert np.isnan(directions[status == 0]).all()
    cropped, cropped_status, _ = ray_map(metric, camera, samples=2, window=(2, 8, 1, 7))
    np.testing.assert_allclose(cropped, directions[4:16, 2:14], equal_nan=True)
    np.testing.assert_array_equal(cropped_status, status[4:16, 2:14])
    assert metric.shadow_angle(800) < metric.shadow_angle(400)
    assert np.tan(metric.shadow_angle(400))/np.tan(np.deg2rad(16)) > np.tan(metric.shadow_angle(400))/np.tan(np.deg2rad(30))
    with pytest.raises(ValueError):
        orbit(metric.shadow_angle(400)*0.9, 400)
