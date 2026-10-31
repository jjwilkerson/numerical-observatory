from dataclasses import asdict
import json

import numpy as np
from PIL import Image
import pytest
from scipy.optimize import brentq

from numerical_observatory.camera import Camera
from numerical_observatory.metric import Schwarzschild
from numerical_observatory.disk import make_disk
from numerical_observatory.disk_validation import quadrature_hit
from numerical_observatory.geodesic import Integration, trace
from numerical_observatory.orbital_angle import OrbitalAngle, orbital_frame, alignment_rings, AmbiguousAngle
from numerical_observatory.orbital_validation import dense_angle, audit_ray, pixel_momentum
from numerical_observatory.orbital_diagnostics import compute_angles, generate, ring_overlay


def launch(angle, radius=60, outward=False):
    metric = Schwarzschild()
    camera = Camera(position=(0, 0, radius))
    p = camera.momentum(metric, [np.sin(angle), 0, (1 if outward else -1)*np.cos(angle)])
    return metric, camera, p


def settings():
    return Integration(rtol=1e-11, atol=1e-13, escape_radius=110, max_affine=400, max_step=4)


def test_weak_field_straight_line_limit_and_both_legs():
    metric, camera, p = launch(.4, radius=1e6)
    orbit = OrbitalAngle.from_initial(metric, camera.event, p)
    assert orbit.background == pytest.approx(np.pi-.4, abs=2e-5)
    # Flat straight-line geometry, independently solving intersections with r=R/2.
    radius, impact, target = 1e6, 1e6*np.sin(.4), 5e5
    for branch, sign in (("inward", -1), ("outward", 1)):
        affine = radius*np.cos(.4)+sign*np.sqrt(target**2-impact**2)
        expected = np.arctan2(affine*np.sin(.4), radius-affine*np.cos(.4))
        assert orbit.to_radius(target, branch) == pytest.approx(expected, abs=3e-5)


def test_inward_turning_and_outward_segments_against_dense_ode():
    metric, camera, p = launch(.2)
    geometry = OrbitalAngle.from_initial(metric, camera.event, p, eps=1e-12)
    result, sol = trace(metric, camera.event, p, settings(), return_solution=True)
    r = np.linalg.norm(sol.y[1:4], axis=0)
    indices = np.flatnonzero((r[:-1]-20)*(r[1:]-20) < 0)
    assert result.status == "escaped" and len(indices) == 2
    for index, branch in zip(indices, ("inward", "outward")):
        t = brentq(lambda t: np.linalg.norm(sol.sol(t)[1:4])-20, sol.t[index], sol.t[index+1])
        measured = dense_angle(metric, sol, camera.event, p, t)
        assert geometry.to_radius(20, branch) == pytest.approx(measured["angle"], abs=1e-8)
    turn = geometry.turn*metric.mass
    assert geometry.to_radius(turn, "inward") == geometry.to_radius(turn, "outward")
    assert geometry.to_radius(turn*1.00001, "inward") < geometry.phase_turn < geometry.to_radius(turn*1.00001, "outward")
    total = geometry.to_radius(20, "inward")+geometry.to_radius(20, "outward")
    assert total == pytest.approx(2*geometry.phase_turn, abs=1e-12)


def test_outward_launch_and_infinity_tail():
    metric, camera, p = launch(.4, outward=True)
    audit = audit_ray(metric, camera.event, p, settings())
    assert audit["ode_status"] == "escaped"
    assert audit["infinity_tail_radians"] > 0
    assert audit["phi_background_error_radians"] < 1e-9
    geometry = OrbitalAngle.from_initial(metric, camera.event, p)
    assert geometry.to_radius(100, "outward") > 0
    with pytest.raises(ValueError):
        geometry.to_radius(20, "inward")


def test_outward_critical_impact_is_not_an_inward_separatrix():
    metric = Schwarzschild()
    angle = metric.shadow_angle(60)
    _, camera, p = launch(angle, outward=True)
    audit = audit_ray(metric, camera.event, p, settings())
    assert audit["quadrature_outcome"] == audit["ode_status"] == "escaped"
    assert audit["phi_background_error_radians"] < 1e-9


def test_captured_and_exactly_radial_rays():
    metric, camera, p = launch(.04)
    geometry = OrbitalAngle.from_initial(metric, camera.event, p)
    result, sol = trace(metric, camera.event, p, settings(), return_solution=True)
    t = brentq(lambda t: np.linalg.norm(sol.sol(t)[1:4])-6, 0, sol.t[-1])
    assert result.status == "captured" and geometry.background is None
    assert geometry.to_radius(6) == pytest.approx(dense_angle(metric, sol, camera.event, p, t)["angle"], abs=1e-9)
    for outward in (False, True):
        _, camera, p = launch(0, outward=outward)
        radial = OrbitalAngle.from_initial(metric, camera.event, p)
        assert radial.to_radius(100 if outward else 6, "outward" if outward else "inward") == 0
        assert radial.background == (0. if outward else None)
    _, camera, p = launch(1e-13)
    near = OrbitalAngle.from_initial(metric, camera.event, p)
    assert 0 < near.to_radius(6) < 1e-10


@pytest.mark.parametrize("epsilon", [1e-4, 1e-5])
def test_near_critical_and_angles_beyond_two_pi(epsilon):
    metric = Schwarzschild()
    angle = np.arcsin(np.sin(metric.shadow_angle(60))*(1+epsilon))
    _, camera, p = launch(angle)
    tight = Integration(rtol=1e-12, atol=1e-14, escape_radius=110, max_affine=400, max_step=2)
    audit = audit_ray(metric, camera.event, p, tight)
    assert audit["ode_status"] == "escaped"
    assert audit["phi_background_quadrature"] > 2*np.pi
    # Near-critical phase sensitivity magnifies conserved-quantity ODE errors;
    # 1e-6 rad is tested at b/bcrit-1>=1e-5, not asserted arbitrarily close to it.
    assert audit["phi_background_error_radians"] < 1e-6
    assert audit["background_dense_checks"]["unwrap_rate_difference"] < 1e-8


def test_critical_and_insufficient_saved_branch_information():
    critical = 3*np.sqrt(3)
    geometry = OrbitalAngle(1, 60, critical)
    assert geometry.outcome == "unresolved" and geometry.background is None
    assert geometry.to_radius(6) > 0
    with pytest.raises(ValueError):
        geometry.to_radius(3)
    looping = OrbitalAngle(1, 60, critical*(1+1e-5))
    radius = brentq(lambda r: (looping.to_radius(r, "outward")-looping.to_radius(r, "inward"))/2-np.pi,
                    looping.turn, 60)
    phi = looping.to_radius(radius, "inward")
    radial, transverse = np.array([0., 0., 1.]), np.array([1., 0., 0.])
    position = radius*(np.cos(phi)*radial+np.sin(phi)*transverse)
    with pytest.raises(AmbiguousAngle):
        looping.saved_hit(position, radial, transverse)
    with pytest.raises(ValueError, match="match"):
        looping.saved_hit([0, 10, 0], radial, transverse)


def test_captured_continuation_after_physical_emission_has_no_sky_angle():
    metric = Schwarzschild()
    camera = Camera(position=(0, 0, 60), width=640, height=480)
    surface = make_disk(metric, dict(enabled=True, inclination_degrees=85)).surface
    p = pixel_momentum(camera, metric, 319, 245)
    audit = audit_ray(metric, camera.event, p, settings(), surface)
    assert audit["crossing"] == 1 and audit["ode_status"] == "captured"
    assert audit["phi_emit_ode"] > 0
    assert "phi_background_ode" not in audit
    assert OrbitalAngle.from_initial(metric, camera.event, p).background is None


def test_partial_quadrature_tolerance_convergence():
    values = []
    for eps in (1e-8, 1e-10, 1e-12):
        geometry = OrbitalAngle(1, 60, 3*np.sqrt(3)*(1+2e-4), eps=eps)
        values.append([geometry.to_radius(r, leg) for r in (6., 12.) for leg in ("inward", "outward")])
    np.testing.assert_allclose(values[:-1], np.broadcast_to(values[-1], (2, 4)), atol=3e-9, rtol=0)


@pytest.mark.parametrize("x,y,crossing", [(200, 245, 1), (310, 192, 2), (284, 272, 3), (285, 272, 4)])
def test_first_to_fourth_emitting_crossings_against_ode(x, y, crossing):
    metric = Schwarzschild()
    camera = Camera(position=(0, 0, 60), width=640, height=480)
    surface = make_disk(metric, dict(enabled=True, inclination_degrees=85)).surface
    audit = audit_ray(metric, camera.event, pixel_momentum(camera, metric, x, y), settings(), surface)
    assert audit["crossing"] == crossing
    assert audit["phi_emit_error_radians"] < 1e-7
    assert audit["phi_background_error_radians"] < 1e-6
    assert audit["phi_background_ode"] > audit["phi_emit_ode"]
    if crossing == 3:
        assert audit["phi_emit_ode"] > 2*np.pi


@pytest.fixture(scope="module")
def rings():
    return alignment_rings(Schwarzschild(), Camera(position=(0, 0, 60), width=640, height=480))


def test_ring_roots_finite_camera_and_precise_projection(rings):
    assert rings["center_xy"] == [319.5, 239.5]
    assert rings["shadow"]["radius_pixels"] == pytest.approx(47.36523115256, abs=1e-10)
    assert rings["rings"][1]["separation_from_shadow_pixels"] == pytest.approx(1.344013698864, abs=1e-9)
    camera, metric = Camera(position=(0, 0, 60), width=640, height=480), Schwarzschild()
    for ring in rings["rings"]:
        p = pixel_momentum(camera, metric, 319.5+ring["radius_pixels"], 239.5)
        measured = OrbitalAngle.from_initial(metric, camera.event, p, eps=1e-12)
        assert measured.background == pytest.approx(ring["phase_pi"]*np.pi, abs=2e-8)
        assert ring["radius_convergence_span_pixels"] < 1e-8
    other = alignment_rings(metric, Camera(position=(0, 0, 400), width=640, height=480), orders=(2,))
    assert other["rings"][0]["radius_pixels"]/other["shadow"]["radius_pixels"] != pytest.approx(
        rings["rings"][1]["radius_pixels"]/rings["shadow"]["radius_pixels"], abs=1e-5)


@pytest.mark.parametrize("order", [1, 2, 3])
def test_alignment_rings_against_independent_ode(rings, order):
    ring = rings["rings"][order-1]
    metric, camera, p = launch(np.deg2rad(ring["angle_degrees"]))
    audit = audit_ray(metric, camera.event, p, settings())
    assert audit["phi_background_ode"] == pytest.approx(order*np.pi, abs=1e-7)


def small_saved():
    camera = Camera(position=(0, 0, 60), width=9, height=7)
    metric = Schwarzschild()
    surface = make_disk(metric, dict(enabled=True, inclination_degrees=85)).surface
    status = np.ones((7, 9), np.uint8)
    status[3, 4] = 0; status[0, 0] = 2; status[0, 1] = 99
    position, crossing = np.full((7, 9, 3), np.nan), np.zeros((7, 9), np.uint16)
    hit = quadrature_hit(metric, camera.event, pixel_momentum(camera, metric, 5, 3), surface)
    assert hit is not None
    status[3, 5], position[3, 5], crossing[3, 5] = 3, hit.event[1:4], hit.crossing_index
    raw = dict(status=status, hit_position=position, hit_radius=np.linalg.norm(position, axis=-1), crossing_index=crossing, samples=np.array(1))
    metadata = dict(camera=asdict(camera), mass=1, geometry=asdict(surface))
    return metadata, raw


def test_maps_do_not_infer_angles_from_crossing_count_and_label_continuation():
    metadata, raw = small_saved()
    arrays, _ = compute_angles(metadata, raw)
    assert arrays["phi_emit_valid"].sum() == 1
    assert arrays["hypothetical_transparent_disk"][3, 5]
    assert not arrays["physically_visible_background"][3, 5]
    assert not arrays["phi_background_valid"][3, 4]
    assert not arrays["phi_background_valid"][0, :2].any()
    assert arrays["background_status"][0, 0] == 2 and arrays["background_status"][0, 1] == 99
    assert arrays["phi_background_radians"][3, 5] > arrays["phi_emit_radians"][3, 5]
    raw["crossing_index"][3, 5] = 100
    different, _ = compute_angles(metadata, raw)
    np.testing.assert_array_equal(arrays["phi_emit_radians"], different["phi_emit_radians"])
    raw.pop("hit_position")
    missing, _ = compute_angles(metadata, raw)
    assert not missing["phi_emit_valid"].any()


def test_overlays_keep_pixel_center_and_crops_match_exactly(tmp_path):
    fake = dict(center_xy=[63.5, 47.5], shadow=dict(radius_pixels=3),
                rings=[dict(phase_pi=1, radius_pixels=25), dict(phase_pi=2, radius_pixels=15), dict(phase_pi=3, radius_pixels=3.05)])
    image = Image.new("RGB", (128, 96), (20, 20, 20))
    overlay = np.asarray(ring_overlay(image, fake)).astype(float)
    magenta = (overlay[..., 0] > 1.3*overlay[..., 1]) & (overlay[..., 2] > 1.3*overlay[..., 1])
    y, x = np.where(magenta)
    assert x.mean() == pytest.approx(63.5, abs=.15) and y.mean() == pytest.approx(47.5, abs=.15)
    assert np.max(abs(np.hypot(x-63.5, y-47.5)-15)) < 2.5
    assert image.getpixel((64, 48)) == (20, 20, 20)
    metadata, raw = small_saved()
    source = tmp_path/"source"; source.mkdir()
    (source/"diagnostics.json").write_text(json.dumps(metadata), encoding="utf-8")
    np.savez_compressed(source/"disk_rays.npz", **raw)
    Image.new("RGB", (9, 7), (70, 60, 50)).save(source/"shadow.png")
    out = tmp_path/"angles"
    report = generate(source, out, features=None, bounds=(2, 1, 7, 6), scale=2)
    assert report["crop_bounds_xyxy_exclusive"] == [2, 1, 7, 6]
    for name in ("visible_alignment_rings", "crossing_alignment_rings", "phi_emit", "phi_background"):
        with Image.open(out/f"{name}_pixels.png") as full, Image.open(out/f"crops/{name}_pixels.png") as crop:
            np.testing.assert_array_equal(crop, full.crop((2, 1, 7, 6)).resize((10, 10), Image.Resampling.NEAREST))
