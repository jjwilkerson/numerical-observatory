from dataclasses import replace
import multiprocessing as mp
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image
import pytest

from numerical_observatory.background import CelestialGrid, Starfield
from numerical_observatory.camera import Camera
from numerical_observatory.disk import make_disk
from numerical_observatory.geodesic import Integration
from numerical_observatory.metric import Schwarzschild
from numerical_observatory.parallel import resolve_workers, cpu_topology
from numerical_observatory.render import render
from numerical_observatory.surfaces import ThinAnnulus


class BrokenSurface(ThinAnnulus):
    def first_hit(self, solution):
        raise ArithmeticError("deliberate worker failure")


class CrashingSurface(ThinAnnulus):
    def first_hit(self, solution):
        os._exit(23)


def interrupt_progress(done, total):
    raise KeyboardInterrupt()


def compare_reports(a, b):
    ignored = {"elapsed_seconds", "execution"}
    assert {k: v for k, v in a.items() if k not in ignored} == {k: v for k, v in b.items() if k not in ignored}


@pytest.mark.parametrize("palette,background,samples,symmetry,method", [
    ("thermal", "starfield", 1, True, "ode"),
    ("gray", "grid", 1, False, "ode"),
    ("blackbody", "grid", 2, True, "quadrature"),
    (None, "starfield", 1, False, "ode"),
    (None, "grid", 2, True, "ode"),
    (None, "starfield", 2, True, "quadrature"),
])
def test_serial_parallel_exact_equality(tmp_path, palette, background, samples, symmetry, method):
    metric = Schwarzschild()
    camera = Camera(position=(0, 0, 60), width=8, height=6)
    settings = Integration(escape_radius=110, max_affine=400, max_step=4)
    sky = Starfield() if background == "starfield" else CelestialGrid()
    disk = None if palette is None else make_disk(metric, dict(enabled=True, inclination_degrees=85,
                                                               display={"palette": palette}))
    reports = []
    for workers in (1, 2, 3):
        progress = []
        report = render(metric, camera, settings, tmp_path/str(workers), background=sky,
                        disk=disk, samples=samples, method=method, use_symmetry=symmetry, workers=workers,
                        progress=lambda completed, total: progress.append((completed, total)))
        reports.append(report)
        assert progress[-1] == (camera.height*samples, camera.height*samples)
        assert [x[0] for x in progress] == sorted(set(x[0] for x in progress))
        with Image.open(tmp_path/"1/shadow.png") as a, Image.open(tmp_path/str(workers)/"shadow.png") as b:
            np.testing.assert_array_equal(a, b)
        name = "disk_rays.npz" if disk else "rays.npz"
        with np.load(tmp_path/"1"/name) as a, np.load(tmp_path/str(workers)/name) as b:
            assert set(a.files) == set(b.files)
            for key in a.files:
                np.testing.assert_array_equal(a[key], b[key], err_msg=key)
        compare_reports(reports[0], report)
        if workers > 1:
            assert report["execution"]["start_method"] == "spawn"
            # Windows CPU accounting can round sub-clock-tick quadrature to 0.
            assert report["execution"]["worker_cpu_seconds"] >= 0
            assert len(report["execution"]["worker_peak_memory_bytes"]) == report["execution"]["workers"]
    if method == "ode" or disk:
        if symmetry:
            yy, xx = np.mgrid[:camera.height*samples, :camera.width*samples]
            expected = len(np.unique((2*xx+1-camera.width*samples)**2+(camera.height*samples-2*yy-1)**2))
        else:
            expected = camera.width*camera.height*samples**2
        assert all(report["integrated_rays"] == expected for report in reports)


@pytest.mark.parametrize("surface,match", [(BrokenSurface, "deliberate worker failure"), (CrashingSurface, "terminated abruptly")])
def test_worker_failure_cleans_up_and_does_not_publish(tmp_path, surface, match):
    metric = Schwarzschild()
    scene = make_disk(metric, {"enabled": True})
    scene = replace(scene, surface=surface(6, 20, scene.surface.normal))
    before = {child.pid for child in mp.active_children()}
    output = tmp_path/"failed"
    with pytest.raises(RuntimeError, match=match):
        render(metric, Camera(position=(0, 0, 60), width=8, height=6),
               Integration(escape_radius=110, max_affine=400), output,
               background=CelestialGrid(), disk=scene, workers=2)
    assert not output.exists()
    assert {child.pid for child in mp.active_children()} == before
    assert not list(tmp_path.glob(".observatory-*"))


def test_interrupt_cleans_up_and_preserves_previous_output(tmp_path):
    output = tmp_path/"previous"
    output.mkdir()
    (output/"diagnostics.json").write_text("previous successful result", encoding="utf-8")
    before = {child.pid for child in mp.active_children()}
    with pytest.raises(KeyboardInterrupt):
        render(Schwarzschild(), Camera(width=8, height=6), Integration(), output,
               progress=interrupt_progress, background=CelestialGrid(), workers=2)
    assert (output/"diagnostics.json").read_text(encoding="utf-8") == "previous successful result"
    assert not (output/"shadow.png").exists()
    assert {child.pid for child in mp.active_children()} == before


def test_worker_selection_and_invalid_counts(monkeypatch):
    import numerical_observatory.parallel as parallel
    for value in (-1, 1.5, True, "5"):
        with pytest.raises(ValueError, match="workers"):
            resolve_workers(value)
    assert resolve_workers(1) == 1
    assert resolve_workers(20) == 20
    monkeypatch.setattr(parallel, "cpu_topology", lambda: dict(physical=10, logical=20))
    assert resolve_workers(0) == 10
    monkeypatch.setattr(parallel, "cpu_topology", lambda: dict(physical=None, logical=7))
    assert resolve_workers(0) == 4


def test_cli_spawn_entrypoint(tmp_path):
    result = subprocess.run([sys.executable, "-m", "numerical_observatory", "--workers", "2",
                             "--background", "grid", "--width", "4", "--height", "3",
                             "--output", str(tmp_path/"cli")], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    assert "estimated remaining" in result.stdout
    assert (tmp_path/"cli/shadow.png").exists()


def test_interruption_while_waiting_stops_running_workers(tmp_path, monkeypatch):
    import numerical_observatory.parallel as parallel
    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt()
    monkeypatch.setattr(parallel, "wait", interrupted)
    before = {child.pid for child in mp.active_children()}
    with pytest.raises(KeyboardInterrupt):
        render(Schwarzschild(), Camera(width=8, height=6), Integration(), tmp_path/"interrupted",
               background=CelestialGrid(), workers=2)
    assert {child.pid for child in mp.active_children()} == before
    assert not (tmp_path/"interrupted").exists()


def test_process_creation_error_is_actionable(tmp_path, monkeypatch):
    import numerical_observatory.parallel as parallel
    def unavailable(*args, **kwargs):
        raise OSError("process quota exceeded")
    monkeypatch.setattr(parallel, "ProcessPoolExecutor", unavailable)
    with pytest.raises(RuntimeError, match="process quota exceeded"):
        render(Schwarzschild(), Camera(width=8, height=6), Integration(), tmp_path/"unavailable",
               background=CelestialGrid(), workers=2)
    assert not (tmp_path/"unavailable").exists()


def test_off_center_camera_uses_general_parallel_path(tmp_path):
    metric = Schwarzschild()
    camera = Camera(position=(0, 0, 60), target=(10, 0, 0), width=10, height=8)
    settings = Integration(escape_radius=110, max_affine=400, max_step=4)
    reports = [render(metric, camera, settings, tmp_path/str(count), workers=count,
                      background=CelestialGrid()) for count in (1, 2)]
    compare_reports(*reports)
    assert not reports[1]["rotational_symmetry_reuse"]
    assert reports[1]["integrated_rays"] == 80
    with np.load(tmp_path/"1/rays.npz") as a, np.load(tmp_path/"2/rays.npz") as b:
        for key in a.files:
            np.testing.assert_array_equal(a[key], b[key])
