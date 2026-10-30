import json

import numpy as np
from PIL import Image
import pytest

from numerical_observatory.camera import Camera
from numerical_observatory.cli import load_configuration
from numerical_observatory.geodesic import Integration
from numerical_observatory.metric import Schwarzschild
from numerical_observatory.render import render
from numerical_observatory.validation import orbit_direction
from test_physics import angular_ray


@pytest.mark.parametrize("angle", [0.26, 0.4, 0.7])
def test_independent_orbit_quadrature(angle):
    result = angular_ray(angle, Integration(rtol=1e-11, atol=1e-13))
    assert result.status == "escaped"
    np.testing.assert_allclose(result.direction, orbit_direction(angle), atol=1e-8)


def test_render_files_and_counts(tmp_path):
    report = render(Schwarzschild(), Camera(width=5, height=3), Integration(), tmp_path)
    assert sum(report["counts"].values()) == 15
    assert report["counts"]["captured"] > 0
    assert report["counts"]["escaped"] > 0
    assert report["counts"]["unresolved"] == 0
    with Image.open(tmp_path / "shadow.png") as image:
        assert image.size == (5, 3)
        assert image.getpixel((2, 1)) == (0, 0, 0)
    assert json.loads((tmp_path / "diagnostics.json").read_text())["counts"] == report["counts"]
    with np.load(tmp_path / "rays.npz") as rays:
        assert rays["directions"].shape == (3, 5, 3)
        assert np.count_nonzero(rays["status"] == 1) == report["counts"]["escaped"]
        np.testing.assert_allclose(np.linalg.norm(rays["directions"][rays["status"] == 1], axis=1), 1)


def test_configuration_mass_scaling(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text("mass = 2.0\n[camera]\nwidth = 3\nheight = 3\n")
    metric, camera, settings = load_configuration(path)
    assert metric.mass == 2 and camera.position == (0, 0, 40)
    assert settings.capture_radius == 3.98
    from numerical_observatory.geodesic import trace
    direction = [np.sin(0.4), 0, -np.cos(0.4)]
    scaled = trace(metric, camera.event, camera.momentum(metric, direction), settings)
    np.testing.assert_allclose(scaled.direction, angular_ray(0.4).direction, atol=1e-7)


def test_invalid_inputs():
    with pytest.raises(ValueError):
        Camera(up=(0, 0, 1))
    with pytest.raises(ValueError):
        Camera(fov_degrees=180)
    with pytest.raises(ValueError):
        Camera(position=(0, 0, 1)).tetrad(Schwarzschild())
    with pytest.raises(ValueError):
        Integration(rtol=-1)
    with pytest.raises(ValueError):
        Schwarzschild(mass=0)
