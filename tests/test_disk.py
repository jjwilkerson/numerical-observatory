from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.constants import sigma

from numerical_observatory.camera import Camera
from numerical_observatory.disk import make_disk
from numerical_observatory.emission import PowerLawBlackbody, frequency_shift, photon_frequency, planck_nu
from numerical_observatory.emitter_motion import SchwarzschildCircular
from numerical_observatory.geodesic import Integration, trace
from numerical_observatory.metric import Schwarzschild
from numerical_observatory.surfaces import ThinAnnulus, CoplanarRay
from numerical_observatory.disk_validation import quadrature_hit
from numerical_observatory.background import CelestialGrid
from numerical_observatory.render import render
from numerical_observatory.surface_render import SurfacePathCache
from numerical_observatory.geodesic import summarize_solution


def test_annulus_multiple_crossings_and_first_valid():
    surface = ThinAnnulus(3, 5, (0, 1, 0))
    def dense(t):
        t = np.asarray(t)
        y = np.zeros((8,)+t.shape)
        y[0], y[1], y[2] = -t, 10-t, np.sin(t)
        y[4] = 1
        return y
    solution = SimpleNamespace(t=np.linspace(0, 10, 41), sol=dense)
    hits = surface.crossings(solution)
    np.testing.assert_allclose([h.affine for h in hits], [np.pi, 2*np.pi, 3*np.pi], atol=1e-10)
    hit = surface.first_hit(solution)
    assert hit.crossing_index == 2 and hit.affine == pytest.approx(2*np.pi)
    assert not surface.contains(hits[0].event) and surface.contains(hit.event)


def test_plane_normal_and_coplanar_handling():
    surface = ThinAnnulus(6, 20, (0, 0, 5))
    assert surface.normal == (0, 0, 1)
    solution = SimpleNamespace(t=np.array([0., 1.]), sol=lambda t: np.array([0*t, 10+0*t, 0*t, 0*t, 1+0*t, 0*t, 0*t, 0*t]))
    with pytest.raises(CoplanarRay):
        surface.first_hit(solution)


@pytest.mark.parametrize("radius", [3.5, 6, 12, 30])
def test_circular_four_velocity_and_local_speed(radius):
    metric = Schwarzschild()
    motion = SchwarzschildCircular(metric, (0, 0, 1))
    event = np.array([0., radius, 0., 0.])
    u = motion.four_velocity(event)
    assert u @ metric.covariant(event) @ u == pytest.approx(-1, abs=2e-14)
    assert u[2]/(radius*u[0]) == pytest.approx(np.sqrt(1/radius**3))
    gamma = np.sqrt(1-2/radius)*u[0]
    assert 1-1/gamma**2 == pytest.approx(1/(radius-2))


def test_disk_defaults_disabled_and_orbit_limits():
    metric = Schwarzschild()
    assert make_disk(metric) is None
    assert make_disk(metric, {"enabled": False}) is None
    assert make_disk(metric, {"enabled": True}).surface.inner_radius == 6
    assert make_disk(metric, {"enabled": True, "inner_radius": 4}).surface.inner_radius == 4
    with pytest.raises(ValueError):
        make_disk(metric, {"enabled": True, "inner_radius": 3})
    with pytest.raises(ValueError):
        SchwarzschildCircular(metric, (0, 0, 1)).four_velocity([0, 3, 0, 0])


def test_frequency_sign_stationary_and_orbiting():
    metric = Schwarzschild()
    ro, re = 60., 8.
    obs = Camera(position=(0, 0, ro)).tetrad(metric)[0]
    emitter_event = np.array([0., re, 0., 0.])
    static = np.array([1/np.sqrt(1-2/re), 0, 0, 0])
    observer_p = np.array([1., 0, 0, 1.])
    emitter_p = np.array([1., 0, 0, 0])
    expected = np.sqrt((1-2/re)/(1-2/ro))
    assert frequency_shift(observer_p, obs, emitter_p, static) == pytest.approx(expected)
    assert frequency_shift(-observer_p, obs, -emitter_p, static, past_directed=False) == pytest.approx(expected)
    with pytest.raises(ValueError):
        photon_frequency(-observer_p, obs)
    orbiting = SchwarzschildCircular(metric, (0, 0, 1)).four_velocity(emitter_event)
    assert frequency_shift(observer_p, obs, emitter_p, orbiting) == pytest.approx(np.sqrt((1-3/re)/(1-2/ro)))
    approaching = emitter_p.copy(); approaching[2] = -0.5
    receding = emitter_p.copy(); receding[2] = 0.5
    assert frequency_shift(observer_p, obs, approaching, orbiting) > frequency_shift(observer_p, obs, receding, orbiting)


def test_intensity_invariance_and_bolometric_factor():
    emission = PowerLawBlackbody(6, temperature_inner=10000)
    event = np.array([0, 6, 0, 0])
    assert emission.bolometric(event) == pytest.approx(sigma/np.pi*10000**4)
    for g in (0.5, 1., 1.5):
        assert emission.observed_bolometric(event, g)/emission.bolometric(event) == pytest.approx(g**4)
        nu = 5e14
        value = emission.observed_specific(event, g, nu)
        assert value/nu**3 == pytest.approx(planck_nu(nu/g, 10000)/(nu/g)**3)
        assert value == pytest.approx(planck_nu(nu, g*10000))


def test_surface_before_capture_obscures_background():
    # A ray that would fall into the hole hits the foreground disk first.
    metric = Schwarzschild()
    camera = Camera(position=(0, 0, 60))
    scene = make_disk(metric, {"enabled": True, "inclination_degrees": 85})
    settings = Integration(escape_radius=100, max_affine=400, max_step=3)
    p = camera.momentum(metric, [0.02, -0.04, -1])
    assert trace(metric, camera.event, p, settings).status == "captured"
    result, solution = trace(metric, camera.event, p, settings, surface=scene.surface, return_solution=True)
    assert result.status == "emitted" and result.direction is None
    assert scene.surface.contains(result.hit.event)
    assert abs(scene.surface.level(result.hit.event)) < 1e-9
    assert abs(result.hit.momentum @ metric.inverse(result.hit.event) @ result.hit.momentum) < 1e-6
    assert solution.t[-1] == result.hit.affine
    with pytest.raises(ValueError):
        solution.sol(result.hit.affine+1)


def test_disabled_disk_ray_results_unchanged():
    metric = Schwarzschild()
    camera = Camera()
    p = camera.momentum(metric, [0.4, 0, -1])
    a = trace(metric, camera.event, p, Integration())
    b = trace(metric, camera.event, p, Integration(), surface=None)
    assert a.status == b.status
    np.testing.assert_array_equal(a.direction, b.direction)


def test_disabled_disk_image_is_bitwise_compatible(tmp_path):
    metric = Schwarzschild()
    camera = Camera(width=8, height=6)
    for name, scene in (("old", None), ("disabled", make_disk(metric, {"enabled": False}))):
        render(metric, camera, Integration(), tmp_path/name, background=CelestialGrid(),
               method="quadrature", disk=scene)
    assert (tmp_path/"old/shadow.png").read_bytes() == (tmp_path/"disabled/shadow.png").read_bytes()


def test_opaque_dark_surface_replaces_background_and_forces_ode(tmp_path):
    from PIL import Image
    metric = Schwarzschild()
    camera = Camera(position=(0, 0, 60), target=(9, -9, 0), width=1, height=1)
    scene = make_disk(metric, {"enabled": True, "emissivity_scale": 0})
    settings = Integration(escape_radius=110, max_affine=400, max_step=4)
    report = render(metric, camera, settings, tmp_path, background=CelestialGrid(),
                    disk=scene, method="quadrature")
    assert report["method"] == "ode_surface" and not report["quadrature_used"]
    assert report["counts"]["emitted"] == 1
    with Image.open(tmp_path/"shadow.png") as image:
        assert image.getpixel((0, 0)) == (0, 0, 0)


@pytest.mark.parametrize("inclination,row,col", [(60, 7, 19), (85, 16, 16)])
def test_disk_quadrature_ode_and_rotated_geometry_agree(inclination, row, col):
    metric = Schwarzschild()
    camera = Camera(position=(0, 0, 60), width=40, height=30)
    p = next(p for r, c, p in camera.rays(metric) if (r, c) == (row, col))
    scene = make_disk(metric, {"enabled": True, "inclination_degrees": inclination})
    settings = Integration(rtol=1e-11, atol=1e-13, escape_radius=110, max_affine=400, max_step=2)
    reference = quadrature_hit(metric, camera.event, p, scene.surface)
    assert reference is not None
    direct = trace(metric, camera.event, p, settings, surface=scene.surface)
    cached = SurfacePathCache(metric, camera, settings)
    reused = summarize_solution(metric, camera.event, p, settings,
                                cached.solution(row, col, p), surface=scene.surface)
    for result in (direct, reused):
        assert result.status == "emitted"
        assert result.hit.crossing_index == reference.crossing_index
        np.testing.assert_allclose(result.hit.event, reference.event, atol=1e-7)
        np.testing.assert_allclose(result.hit.momentum, reference.momentum, atol=1e-8)
    if inclination == 85:
        assert reference.crossing_index == 2


def test_circular_motion_satisfies_geodesic_acceleration():
    metric = Schwarzschild()
    event = np.array([0., 8., 0., 0.])
    u = SchwarzschildCircular(metric, (0, 0, 1)).four_velocity(event)
    g, inverse = metric.covariant(event), metric.inverse(event)
    dg = -np.einsum("mi,aij,jn->amn", g, metric.inverse_derivative(event), g)
    gamma = np.zeros((4, 4, 4))
    for a in range(4):
        for b in range(4):
            for c in range(4):
                gamma[a, b, c] = sum(inverse[a, d]*(dg[b, d, c]+dg[c, d, b]-dg[d, b, c])/2 for d in range(4))
    derivative = np.r_[0., -(u[0]**2/8**3)*event[1:]]
    np.testing.assert_allclose(derivative+np.einsum("abc,b,c->a", gamma, u, u), 0, atol=1e-14)


def test_fourth_crossing_after_winding_and_tolerance_convergence():
    metric = Schwarzschild()
    camera = Camera(position=(0, 0, 60))
    scene = make_disk(metric, {"enabled": True})
    angle = np.arcsin(np.sin(metric.shadow_angle(60))*(1+4e-5))
    p = camera.momentum(metric, [0, np.sin(angle), -np.cos(angle)])
    reference = quadrature_hit(metric, camera.event, p, scene.surface)
    assert reference is not None and reference.crossing_index == 4
    errors = []
    for tol in (1e-7, 1e-11):
        settings = Integration(rtol=tol, atol=tol/100, escape_radius=110, max_affine=400, max_step=2)
        result = trace(metric, camera.event, p, settings, surface=scene.surface)
        assert result.status == "emitted" and result.hit.crossing_index == 4
        errors.append(np.linalg.norm(result.hit.event[1:]-reference.event[1:]))
    assert errors[1] < 1e-5 and errors[1] < errors[0]/20


def test_doppler_bolometric_asymmetry_and_rotation_reversal():
    metric = Schwarzschild()
    camera = Camera(position=(0, 0, 60))
    settings = Integration(rtol=1e-11, atol=1e-13, escape_radius=110, max_affine=400, max_step=2)
    ratios = []
    for rotation in (1, -1):
        scene = make_disk(metric, {"enabled": True, "rotation": rotation})
        intensities, shifts = [], []
        for x in (-0.15, 0.15):
            p = camera.momentum(metric, [x, -0.15, -1])
            result = trace(metric, camera.event, p, settings, surface=scene.surface)
            assert result.status == "emitted"
            hit = result.hit
            g = frequency_shift(p, camera.tetrad(metric)[0], hit.momentum, scene.motion.four_velocity(hit.event))
            shifts.append(g)
            intensities.append(scene.emission.observed_bolometric(hit.event, g))
        assert intensities[0]/intensities[1] == pytest.approx((shifts[0]/shifts[1])**4, rel=1e-8)
        ratios.append(intensities[0]/intensities[1])
    assert ratios[0] > 1 and ratios[1] < 1
    assert ratios[0]*ratios[1] == pytest.approx(1, rel=1e-8)
