import numpy as np
import pytest

from numerical_observatory.background import Starfield
from numerical_observatory.camera import Camera
from numerical_observatory.geodesic import Integration, trace
from numerical_observatory.metric import Schwarzschild
from numerical_observatory.symmetry import CenteredRayCache


def test_sky_map_seam_poles_and_exposure():
    sky = Starfield(longitude_degrees=0, latitude_degrees=0, roll_degrees=0)
    assert sky.texture.shape == (2048, 4096, 3)
    np.testing.assert_allclose(np.linalg.det(sky.rotation), 1, atol=1e-14)
    assert sky.description["accretion_disk"] is False
    directions = np.array([[1e-12, 0, 1], [-1e-12, 0, 1], [0, 1, 0], [0, -1, 0]])
    rgb = sky.color(directions)
    assert np.isfinite(rgb).all() and np.all((rgb >= 0) & (rgb <= 1))
    np.testing.assert_allclose(rgb[0], rgb[1], atol=1e-8)
    sky.exposure = 1
    assert np.all(sky.color(directions) <= rgb+1e-12)


@pytest.mark.parametrize("position,dimensions", [((0, 0, 20), (8, 6)), ((8, 4, 15), (7, 5))])
def test_rotational_reuse_matches_direct_geodesics(position, dimensions):
    metric = Schwarzschild()
    camera = Camera(position=position, width=dimensions[0], height=dimensions[1])
    settings = Integration(rtol=1e-10, atol=1e-12)
    cached = CenteredRayCache(metric, camera, settings)
    assert cached.enabled
    for row, col, momentum in camera.rays(metric):
        a = cached.ray(row, col, momentum)
        b = trace(metric, camera.event, momentum, settings)
        assert a.status == b.status
        if a.direction is not None:
            np.testing.assert_allclose(a.direction, b.direction, atol=2e-8)
    assert cached.integrated_rays < camera.width*camera.height/2


def test_off_center_disables_symmetry():
    cache = CenteredRayCache(Schwarzschild(), Camera(target=(1, 0, 0)), Integration())
    assert not cache.enabled
