from dataclasses import asdict, replace
import hashlib
import json
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from numerical_observatory.background import CelestialGrid
from numerical_observatory.camera import Camera
from numerical_observatory.diagnostic_zoom import window_camera
from numerical_observatory.disk import make_disk
from numerical_observatory.disk_layers import (collect, layer_grid, make_views, trace_to_directory,
                                               first_hits, compare_opaque)
from numerical_observatory.emission import frequency_shift
from numerical_observatory.geodesic import Integration, trace
from numerical_observatory.layer_tracing import (LayerScene, pack_crossings, collect_solution,
                                                layer_task, CROSSING_DTYPE)
from numerical_observatory.metric import Schwarzschild
from numerical_observatory.orbital_validation import pixel_momentum
from numerical_observatory.render import render
from numerical_observatory.surfaces import SurfaceHit, ThinAnnulus


def context(width=640, height=480):
    metric = Schwarzschild()
    camera = Camera(position=(0, 0, 60), width=width, height=height)
    settings = Integration(rtol=1e-11, atol=1e-13, escape_radius=110, max_affine=400, max_step=4)
    disk = make_disk(metric, dict(enabled=True, inclination_degrees=85, display=dict(palette="blackbody")))
    return metric, camera, settings, disk


def packed(radii):
    metric = Schwarzschild()
    camera = Camera(position=(0, 0, 60))
    initial = camera.momentum(metric, [0, 0, -1])
    disk = make_disk(metric, dict(enabled=True, inclination_degrees=0))
    observer = camera.tetrad(metric)[0]
    hits = [SurfaceHit((i+1)*10., np.array([-(i+1)*10., r, 0, 0]),
                       np.array([initial[0], -initial[0], 0, 0]), i+1) for i, r in enumerate(radii)]
    records = pack_crossings(metric, initial, observer, disk, hits, phases=np.arange(len(hits))*np.pi)
    data = dict(crossings=records, pixel_index=np.zeros(len(records), np.int64),
                ray_offsets=np.array([0, len(records)]), ray_status=np.array([[1]], np.uint8),
                crossing_limit_reached=np.array([[False]]))
    return data, initial[0]


def test_hole_outer_edge_and_plane_vs_emitting_ordinal():
    data, energy = packed([4, 12, 25, 10, 8])
    records = data["crossings"]
    np.testing.assert_array_equal(records["plane_ordinal"], [1, 2, 3, 4, 5])
    np.testing.assert_array_equal(records["region"], [1, 2, 3, 2, 2])
    np.testing.assert_array_equal(records["emitting_ordinal"], [0, 1, 0, 2, 3])
    assert layer_grid(data, 2)["normally_visible"][0, 0]
    assert layer_grid(data, 4)["concealed"][0, 0]
    assert layer_grid(data, 5)["concealed"][0, 0]
    assert not layer_grid(data, 3)["valid"].any()  # no substitution from crossing 4
    assert np.all(np.diff(records["affine"]) > 0)
    assert np.all(np.diff(records["event"][:, 0]) < 0)
    for row in records[records["emission_valid"]]:
        radius = row["radius_M"]
        expected = np.sqrt(1-3/radius)/energy  # radial p is orthogonal to circular velocity
        assert row["g"] == pytest.approx(expected, rel=1e-14)
        from scipy.constants import sigma
        temperature = 10000*(radius/6)**-.75
        assert row["bolometric_intensity"] == pytest.approx(expected**4*sigma/np.pi*temperature**4, rel=1e-14)


def test_earlier_material_conceals_later_material_even_if_dark():
    data, _ = packed([10, 12, 25, 4])
    assert layer_grid(data, 1)["normally_visible"].all()
    assert layer_grid(data, 2)["concealed"].all()
    assert not layer_grid(data, 3)["valid"].any()
    assert not layer_grid(data, 4)["valid"].any()
    assert first_hits(data)[0] == 0
    data["crossings"]["bolometric_intensity"] = 0
    assert layer_grid(data, 2)["concealed"].all()


def test_no_duplicate_crossings_at_step_nodes():
    surface = ThinAnnulus(6, 20, (0, 1, 0))
    def dense(t):
        t = np.asarray(t)
        return np.array([-t, 10+0*t, np.sin(t), 0*t, 1+0*t, -1+0*t, 0*t, 0*t])
    times = np.unique(np.r_[np.linspace(0, 4*np.pi+.1, 150), np.arange(1, 5)*np.pi])
    hits = surface.crossings(SimpleNamespace(t=times, sol=dense))
    assert len(hits) == 4
    np.testing.assert_allclose([hit.affine for hit in hits], np.arange(1, 5)*np.pi, atol=1e-10)
    assert [hit.crossing_index for hit in hits] == [1, 2, 3, 4]


@pytest.mark.parametrize("x,y,regions", [(310, 192, [1, 2]), (319, 310, [3, 2]),
                                        (260, 250, [2, 2]), (379, 250, [2, 2]), (285, 272, [3, 1, 1, 2]),
                                        (275, 245, [2, 1])])
def test_all_crossings_quadrature_versus_ode(x, y, regions):
    metric, camera, settings, disk = context()
    index = y*camera.width+x
    quad, _ = layer_task((metric, camera, settings, disk, 4, "quadrature"), [index])
    ode, _ = layer_task((metric, camera, settings, disk, 4, "ode"), [index])
    a, b = quad[0][1], ode[0][1]
    np.testing.assert_array_equal(a["region"], regions)
    np.testing.assert_array_equal(a["region"], b["region"])
    np.testing.assert_array_equal(a["plane_ordinal"], b["plane_ordinal"])
    np.testing.assert_array_equal(a["emission_valid"], b["emission_valid"])
    np.testing.assert_allclose(a["event"], b["event"], atol=3e-7, rtol=1e-8)
    np.testing.assert_allclose(a["momentum"], b["momentum"], atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(a["affine"], b["affine"], atol=3e-7, rtol=1e-8)
    np.testing.assert_allclose(a["g"], b["g"], atol=1e-8, rtol=1e-8, equal_nan=True)
    valid = a["emission_valid"]
    np.testing.assert_allclose(a["bolometric_intensity"][valid], b["bolometric_intensity"][valid], rtol=2e-7)
    np.testing.assert_allclose(a["phi_radians"], b["phi_radians"], atol=1e-7)
    assert np.all(np.diff(a["affine"]) > 0)
    assert a["null_error"].max() < 1e-12


def test_first_valid_ode_hit_matches_opaque_and_is_not_modified():
    metric, camera, settings, disk = context()
    p = pixel_momentum(camera, metric, 260, 250)
    ordinary = trace(metric, camera.event, p, settings, surface=disk.surface)
    _, sol = trace(metric, camera.event, p, settings, return_solution=True)
    scene = LayerScene(disk.surface, disk.motion, disk.emission, disk.display, camera.tetrad(metric)[0])
    records, *_ = collect_solution(metric, camera.event, p, settings, sol, scene, 4)
    valid = records[records["emission_valid"]]
    assert len(valid) == 2
    np.testing.assert_array_equal(valid[0]["event"], ordinary.hit.event)
    np.testing.assert_array_equal(valid[0]["momentum"], ordinary.hit.momentum)
    assert valid[0]["g"] == frequency_shift(p, scene.motion_observer, ordinary.hit.momentum, disk.motion.four_velocity(ordinary.hit.event))
    assert trace(metric, camera.event, p, settings, surface=disk.surface).hit.crossing_index == ordinary.hit.crossing_index


@pytest.mark.parametrize("method", ["ode", "quadrature"])
def test_worker_counts_are_exact_and_do_not_duplicate_geometry(method):
    metric, camera, settings, disk = context(8, 6)
    baseline, report = collect(metric, camera, settings, disk, 4, 1, method)
    for count in (2, 3):
        result, parallel = collect(metric, camera, settings, disk, 4, count, method)
        for key in baseline:
            assert baseline[key].dtype == result[key].dtype
            assert baseline[key].tobytes() == result[key].tobytes(), key
        assert parallel["work"] == report["work"]
        assert parallel["counts"] == report["counts"]


def test_window_camera_matches_corresponding_full_frame():
    metric, camera, settings, disk = context(16, 12)
    complete, _ = collect(metric, camera, settings, disk, 4, 1, "quadrature")
    window = window_camera(camera, (3, 2, 11, 9), 1)
    cropped, _ = collect(metric, window, settings, disk, 4, 1, "quadrature")
    for row in range(window.height):
        for col in range(window.width):
            full_index = (row+2)*camera.width+col+3
            local = row*window.width+col
            a = complete["crossings"][complete["ray_offsets"][full_index]:complete["ray_offsets"][full_index+1]]
            b = cropped["crossings"][cropped["ray_offsets"][local]:cropped["ray_offsets"][local+1]]
            assert a.tobytes() == b.tobytes()
    np.testing.assert_array_equal(cropped["ray_status"], complete["ray_status"][2:9, 3:11])


def test_recording_limit_does_not_substitute_later_emission():
    metric, camera, settings, disk = context()
    result, _ = layer_task((metric, camera, settings, disk, 2, "quadrature"), [272*640+284])
    records = result[0][1]
    assert len(records) == 2 and not records["emission_valid"].any()
    assert result[0][6]  # an unrecorded third crossing exists


@pytest.mark.parametrize("method", ["ode", "quadrature"])
def test_affine_limit_and_coplanar_rejection(method):
    metric, camera, settings, disk = context(3, 3)
    short = replace(settings, max_affine=.1)
    arrays, _ = collect(metric, camera, short, disk, 4, 1, method)
    assert not len(arrays["crossings"])
    assert np.all(arrays["ray_status"] == 2)
    assert np.all(arrays["ray_reason"] == 2)
    coplanar = make_disk(metric, dict(enabled=True, inclination_degrees=90))
    result, _ = layer_task((metric, camera, settings, coplanar, 4, method), [4])
    assert result[0][2] == 2 and result[0][3] == 4
    assert len(result[0][1]) == 0


def test_repaint_crops_alpha_palettes_and_numerical_data_are_independent(tmp_path, monkeypatch):
    metric, camera, settings, disk = context(32, 24)
    metadata = dict(mass=1, camera=asdict(camera), integration=asdict(settings), geometry=asdict(disk.surface),
                    emission=asdict(disk.emission), display=asdict(disk.display), rotation=1, background=CelestialGrid().description)
    directory = tmp_path/"layers"
    trace_to_directory(metadata, directory, method="quadrature")
    sha = hashlib.sha256((directory/"disk_layers.npz").read_bytes()).hexdigest()
    import numerical_observatory.disk_layers as module
    monkeypatch.setattr(module, "collect", lambda *a, **k: pytest.fail("repaint retraced"))
    reports = []
    for palette in ("thermal", "gray", "blackbody"):
        dest = tmp_path/palette
        reports.append(make_views(directory, dest, alpha=0, bounds=(8, 5, 24, 19), scale=3, palette=palette, features=None))
        with Image.open(dest/"crossing_2.png") as layer, Image.open(dest/"composite.png") as composite:
            np.testing.assert_array_equal(layer, composite)
        with Image.open(dest/"opaque.png") as full, Image.open(dest/"crops/opaque.png") as crop:
            np.testing.assert_array_equal(crop, full.crop((8, 5, 24, 19)).resize(crop.size, Image.Resampling.NEAREST))
        assert (dest/"crossing_1.png").exists() and (dest/"crossing_3.png").exists() and (dest/"crossing_4.png").exists()
    assert len(set(r["concealed_subrays"] for r in reports)) == 1
    assert hashlib.sha256((directory/"disk_layers.npz").read_bytes()).hexdigest() == sha


def test_visible_second_crossing_and_opaque_image_compatibility(tmp_path):
    metric, camera, settings, disk = context(16, 12)
    sky = CelestialGrid()
    before = render(metric, camera, settings, tmp_path/"before", background=sky, disk=disk)
    data, _ = collect(metric, camera, settings, disk, 4, 1, "ode")
    with np.load(tmp_path/"before/disk_rays.npz") as reference:
        report = compare_opaque(data, reference)
    assert report["first_hit_category_differences"] == 0
    assert report["normally_visible_crossing_2_differences"] == 0
    render(metric, camera, settings, tmp_path/"after", background=sky, disk=disk)
    assert (tmp_path/"before/shadow.png").read_bytes() == (tmp_path/"after/shadow.png").read_bytes()


def test_invalid_limits():
    metric, camera, settings, disk = context(3, 3)
    for maximum in (0, -1, True):
        with pytest.raises(ValueError, match="max-crossing"):
            collect(metric, camera, settings, disk, maximum)


@pytest.mark.parametrize("flags", [["--width", "0"], ["--target", "5", "--max-crossing", "4"]])
def test_invalid_cli_options_do_not_launch_a_trace(tmp_path, monkeypatch, flags):
    import sys
    import numerical_observatory.disk_layers as module
    config = tmp_path/"scene.toml"
    config.write_text('[camera]\nwidth=8\nheight=6\n[disk]\nenabled=true\n[background]\nmode="grid"\n', encoding="utf-8")
    monkeypatch.setattr(module, "trace_to_directory", lambda *a, **k: pytest.fail("invalid options started tracing"))
    monkeypatch.setattr(sys, "argv", ["disk_layers", "--config", str(config), "--output", str(tmp_path/"unused")]+flags)
    with pytest.raises(SystemExit) as error:
        module.main()
    assert error.value.code == 2
