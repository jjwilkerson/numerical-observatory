import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from numerical_observatory.disk_diagnostics import (
    extract, disk_basis, azimuth_color, radius_color, colorize, generate, crop_bounds,
)
from numerical_observatory.diagnostic_zoom import window_camera
from numerical_observatory.camera import Camera
from numerical_observatory.metric import Schwarzschild
from numerical_observatory.surfaces import ThinAnnulus


def known_data(normal=(0, 0, 1)):
    normal, a, b = disk_basis(normal)
    angles = np.array([[0, 90, 180, 270], [359.999, .001, 45, 315.]])
    radii = np.array([[6, 8, 12, 20], [10, 10, 15, 18.]])
    positions = 2*radii[..., None]*(np.cos(np.deg2rad(angles))[..., None]*a+
                                   np.sin(np.deg2rad(angles))[..., None]*b)
    data = dict(status=np.full((2, 4), 3, dtype=np.uint8), hit_position=positions,
                hit_radius=2*radii, crossing_index=np.array([[1, 2, 3, 4], [7, 10, 2, 1]]))
    metadata = dict(mass=2., geometry=dict(normal=normal.tolist(), inner_radius=12, outer_radius=40))
    return data, metadata, angles, radii


@pytest.mark.parametrize("normal", [(0, 0, 1), (0, np.sin(np.deg2rad(85)), np.cos(np.deg2rad(85))), (1, 0, 0), (-1, 0, 0)])
def test_exact_radius_azimuth_crossings_and_handedness(normal):
    data, metadata, angles, radii = known_data(normal)
    original = {k: v.copy() for k, v in data.items()}
    arrays, info = extract(data, metadata)
    np.testing.assert_allclose(arrays["radius_m"], radii)
    np.testing.assert_array_equal(arrays["crossing_number"], data["crossing_index"])
    error = (arrays["azimuth_degrees"]-angles+180) % 360-180
    np.testing.assert_allclose(error, 0, atol=1e-12)
    assert all(arrays[key+"_valid"].all() for key in ("radius", "crossing", "azimuth"))
    n, a, b = disk_basis(normal)
    np.testing.assert_allclose(np.cross(a, b), n, atol=1e-14)
    assert info["max_saved_radius_position_disagreement"] < 1e-13
    for key in data:
        np.testing.assert_array_equal(data[key], original[key])


def test_angular_wrap_and_color_continuity():
    np.testing.assert_array_equal(azimuth_color(0), azimuth_color(360))
    np.testing.assert_array_equal(azimuth_color(-90), azimuth_color(270))
    colors = azimuth_color(np.array([359.999, 0, .001])).astype(int)
    assert np.max(np.abs(np.diff(colors, axis=0))) <= 1
    assert azimuth_color(np.zeros((2, 3))).shape == (2, 3, 3)


def test_nonemitting_unresolved_and_invalid_values():
    data, metadata, _, _ = known_data()
    data["status"][0] = [0, 1, 2, 99]
    data["hit_radius"][1, 0] = np.nan
    data["hit_position"][1, 1] = np.nan
    data["crossing_index"][1, 2] = 0
    arrays, info = extract(data, metadata)
    for key in ("radius_valid", "crossing_valid", "azimuth_valid"):
        assert not arrays[key][0].any()
    assert np.isnan(arrays["radius_m"][0]).all()
    assert np.isnan(arrays["azimuth_degrees"][0]).all()
    assert not arrays["crossing_number"][0].any()
    assert not arrays["radius_valid"][1, 0]
    assert not arrays["azimuth_valid"][1, 1]
    assert not arrays["crossing_valid"][1, 2]
    maps = colorize(arrays, 6, 20)
    rgb = np.asarray(maps["crossing"])
    np.testing.assert_array_equal(rgb[0, 0], [0, 0, 0])
    assert not np.array_equal(rgb[0, 1], rgb[0, 2])
    assert info["missing_emitted_counts"]["radius"] == 1


def test_older_data_missing_fields_are_not_inferred():
    data, metadata, _, radii = known_data()
    del data["hit_radius"]
    arrays, info = extract(data, metadata)
    np.testing.assert_allclose(arrays["radius_m"], radii)
    assert info["radius_source"] == "norm(hit_position)"
    del data["hit_position"]
    del data["crossing_index"]
    arrays, info = extract(data, metadata)
    assert len(info["warnings"]) == 3
    assert not arrays["radius_valid"].any() and not arrays["azimuth_valid"].any()
    assert not arrays["crossing_valid"].any()
    assert np.isnan(arrays["azimuth_degrees"]).all()
    assert not arrays["crossing_number"].any()


def test_first_valid_hit_is_preserved_after_earlier_hole_crossing():
    def dense(t):
        t = np.asarray(t)
        result = np.zeros((8,)+t.shape)
        result[1], result[2], result[4] = 10-t, np.sin(t), 1
        return result
    surface = ThinAnnulus(3, 5, (0, 1, 0))
    hit = surface.first_hit(SimpleNamespace(t=np.linspace(0, 10, 101), sol=dense))
    assert hit.crossing_index == 2
    arrays, _ = extract(dict(status=np.array([[3]]), hit_position=hit.event[None, None, 1:],
                             crossing_index=np.array([[hit.crossing_index]])),
                        dict(mass=1, geometry=dict(normal=surface.normal)))
    assert arrays["crossing_number"][0, 0] == 2
    assert arrays["radius_m"][0, 0] == pytest.approx(10-2*np.pi)


@pytest.mark.parametrize("samples", [1, 2])
def test_pixel_positions_and_four_matching_crops(tmp_path, samples):
    source = tmp_path/"input"
    source.mkdir()
    data, metadata, _, _ = known_data()
    # 4x2 output image; with supersampling, retain every distinct subray.
    if samples == 2:
        data = {key: np.repeat(np.repeat(value, 2, axis=0), 2, axis=1) for key, value in data.items()}
    np.savez_compressed(source/"disk_rays.npz", **data, samples=samples)
    (source/"diagnostics.json").write_text(json.dumps(metadata), encoding="utf-8")
    image = np.arange(24, dtype=np.uint8).reshape(2, 4, 3)*10
    Image.fromarray(image).save(source/"shadow.png")
    before = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in source.iterdir()}
    output = tmp_path/"maps"
    report = generate(source, output, bounds=(1, 0, 4, 2), scale=3)
    assert report["crop_bounds_xyxy_exclusive"] == [1, 0, 4, 2]
    assert report["geodesics_retraced"] is False
    for name in ("emission_radius", "crossing_number", "emission_azimuth"):
        with Image.open(output/f"{name}_pixels.png") as full, Image.open(output/f"crops/{name}_pixels.png") as cropped:
            expected = full.crop((samples, 0, 4*samples, 2*samples)).resize((9*samples, 6*samples), Image.Resampling.NEAREST)
            np.testing.assert_array_equal(cropped, expected)
    with Image.open(output/"crops/rendered_pixels.png") as cropped:
        expected = Image.fromarray(image).crop((1, 0, 4, 2)).resize((9*samples, 6*samples), Image.Resampling.NEAREST)
        np.testing.assert_array_equal(cropped, expected)
    assert before == {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in source.iterdir()}
    with np.load(output/"diagnostic_arrays.npz") as arrays:
        np.testing.assert_array_equal(arrays["crossing_number"], data["crossing_index"])
    with np.load(output/"crops/diagnostic_arrays.npz") as arrays:
        np.testing.assert_array_equal(arrays["crossing_number"], data["crossing_index"][:, samples:])
    with pytest.raises(ValueError, match="separate"):
        generate(source, source/"nested")


def test_invalid_geometry_shapes_and_bounds():
    data, metadata, _, _ = known_data()
    with pytest.raises(ValueError, match="mass"):
        extract(data, dict(mass=0))
    data["hit_position"] = np.zeros((2, 3, 3))
    with pytest.raises(ValueError, match="shape"):
        extract(data, metadata)
    with pytest.raises(ValueError, match="within"):
        crop_bounds((-1, 0, 4, 2), 4, 2)
    with pytest.raises(ValueError, match="normal"):
        disk_basis((0, 0, 0))


@pytest.mark.parametrize("density", [1, 3])
def test_optional_window_preserves_full_camera_ray_mapping(density):
    camera = Camera(position=(0, 0, 60), width=8, height=6)
    metric = Schwarzschild()
    window = window_camera(camera, (2, 1, 5, 4), density)
    full = Camera(position=camera.position, width=8*density, height=6*density)
    expected = {(r-density, c-2*density): p for r, c, p in full.rays(metric)
                if density <= r < 4*density and 2*density <= c < 5*density}
    for row, col, momentum in window.rays(metric):
        np.testing.assert_array_equal(momentum, expected[row, col])


def test_existing_serial_parallel_saved_data_give_identical_diagnostics():
    root = Path(__file__).resolve().parents[1]/"examples/parallel_benchmark/disk_moderate"
    if not (root/"1/disk_rays.npz").exists():
        pytest.skip("optional saved benchmark fixture is unavailable")
    extracted = []
    for workers in (1, 5, 10, 20):
        metadata = json.loads((root/str(workers)/"diagnostics.json").read_text(encoding="utf-8"))
        with np.load(root/str(workers)/"disk_rays.npz") as data:
            arrays, _ = extract(data, metadata)
            extracted.append(arrays)
    for arrays in extracted[1:]:
        for key in extracted[0]:
            np.testing.assert_array_equal(arrays[key], extracted[0][key])
