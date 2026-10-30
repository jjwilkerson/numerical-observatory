"""ODE-only opaque-surface rendering with optional exact trajectory rotations."""
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from collections import Counter
import json
import time

import numpy as np

from .background import linear_to_srgb, Starfield
from .diagnostic_render import save_image
from .display import srgb_to_linear
from .emission import frequency_shift
from .geodesic import trace, summarize_solution
from .schwarzschild_orbit import compatible


class SurfacePathCache:
    """Reuse GEOMETRY only, not hits/emission, for a centered Schwarzschild view.

    A tilted disk breaks image symmetry. Every pixel tests its own rotated
    trajectory against that fixed disk. No near/far brightness is mirrored.
    """
    def __init__(self, metric, camera, integration, domain=None, enabled=True):
        self.metric, self.camera, self.integration, self.domain = metric, camera, integration, domain
        self.enabled = enabled and compatible(metric, camera) and domain is None
        self.forward, self.right, self.up = camera.axes()
        self.paths = {}

    def solution(self, row, col, momentum):
        camera = self.camera
        ix, iy = 2*col+1-camera.width, camera.height-2*row-1
        key = ix*ix+iy*iy
        if not self.enabled:
            return trace(self.metric, camera.event, momentum, self.integration,
                         domain=self.domain, return_solution=True)[1]
        if key not in self.paths:
            offset = np.sqrt(key)*np.tan(np.deg2rad(camera.fov_degrees)/2)/camera.width
            canonical = camera.momentum(self.metric, self.forward+offset*self.right)
            self.paths[key] = trace(self.metric, camera.event, canonical, self.integration,
                                    return_solution=True)[1]
        sol = self.paths[key]
        if key == 0:
            return sol
        transverse = (ix*self.right+iy*self.up)/np.sqrt(key)
        radial = -self.forward
        rotation = (np.outer(radial, radial)+np.outer(transverse, self.right)
                    +np.outer(np.cross(radial, transverse), np.cross(radial, self.right)))
        def transform(y):
            result = y.copy()
            result[1:4] = rotation @ y[1:4]
            result[5:8] = rotation @ y[5:8]
            return result
        return SimpleNamespace(t=sol.t, y=transform(sol.y), sol=lambda t: transform(sol.sol(t)),
                               t_events=sol.t_events, success=sol.success, message=sol.message, nfev=sol.nfev)


def render_surface(metric, camera, integration, output, scene, background=None,
                   samples=1, domain=None, progress=None, path_cache=None, requested_method="ode",
                   use_symmetry=True):
    from dataclasses import replace
    if not isinstance(samples, int) or samples < 1:
        raise ValueError("samples must be a positive integer")
    sampled = replace(camera, width=camera.width*samples, height=camera.height*samples)
    observer = camera.tetrad(metric)[0]
    background = Starfield() if background is None else background
    cache = path_cache or SurfacePathCache(metric, sampled, integration, domain, enabled=use_symmetry)
    if (cache.metric is not metric or cache.camera != sampled or cache.integration != integration
            or cache.domain is not domain):
        raise ValueError("path cache does not match metric/camera/integration/domain")
    if not use_symmetry and cache.enabled:
        raise ValueError("supplied path cache enables disabled symmetry reuse")
    if abs(scene.surface.level(camera.event)) < 1e-10 and scene.surface.contains(camera.event):
        raise ValueError("camera cannot lie on the opaque emitting surface")
    h, w = sampled.height, sampled.width
    codes = np.zeros((h, w), dtype=np.uint8)
    shift = np.full((h, w), np.nan)
    intensity = np.zeros((h, w))
    temperature = np.full((h, w), np.nan)
    hit_radius = np.full((h, w), np.nan)
    hit_position = np.full((h, w, 3), np.nan)
    crossing = np.zeros((h, w), dtype=np.uint16)
    directions = np.full((h, w, 3), np.nan)
    rgb = np.zeros((h, w, 3))
    null_max = 0.0
    counts, reasons = Counter(), Counter()
    start = time.perf_counter()
    for row, col, momentum in sampled.rays(metric):
        sol = cache.solution(row, col, momentum)
        result = summarize_solution(metric, sampled.event, momentum, integration, sol, domain=domain,
                                    builtin_domain=cache.domain is None, surface=scene.surface)
        counts[result.status] += 1
        reasons[result.reason] += 1
        null_max = max(null_max, result.max_null_error)
        if result.status == "emitted":
            hit = result.hit
            g = frequency_shift(momentum, observer, hit.momentum, scene.motion.four_velocity(hit.event))
            bolometric = scene.emission.observed_bolometric(hit.event, g)
            observed_temperature = g*scene.emission.temperature(hit.event)
            codes[row, col] = 3
            shift[row, col], intensity[row, col] = g, bolometric
            temperature[row, col] = observed_temperature
            hit_radius[row, col] = np.linalg.norm(hit.event[1:4])
            hit_position[row, col] = hit.event[1:4]
            crossing[row, col] = hit.crossing_index
            rgb[row, col] = scene.display.linear_rgb(bolometric, observed_temperature)
        elif result.status == "escaped":
            codes[row, col] = 1
            directions[row, col] = result.direction
            rgb[row, col] = srgb_to_linear(background.color(result.direction))
        elif result.status == "unresolved":
            codes[row, col] = 2
            rgb[row, col] = (1, 0, 0.7)
        if progress and col == w-1:
            progress(row+1, h)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    image = linear_to_srgb(rgb.reshape(camera.height, samples, camera.width, samples, 3).mean(axis=(1, 3)))
    save_image(output/"shadow.png", image)
    np.savez_compressed(output/"disk_rays.npz", status=codes, g=shift, bolometric_intensity=intensity,
                        observed_temperature=temperature, hit_radius=hit_radius, hit_position=hit_position,
                        crossing_index=crossing, directions=directions, samples=samples)
    emitted = codes == 3
    report = dict(method="ode_surface", requested_method=requested_method,
                  quadrature_used=False, exact_rotational_path_reuse=cache.enabled,
                  camera=asdict(camera), integration=asdict(integration), background=background.description,
                  geometry=asdict(scene.surface), emission=asdict(scene.emission), display=asdict(scene.display),
                  rotation=getattr(scene.motion, "rotation", None), mass=getattr(metric, "mass", None),
                  counts={key: counts[key] for key in ("captured", "escaped", "unresolved", "emitted")},
                  termination_reasons=dict(reasons), max_sampled_null_error=null_max,
                  samples_per_axis=samples, elapsed_seconds=time.perf_counter()-start,
                  count_units="subpixel rays", g_range=[float(shift[emitted].min()), float(shift[emitted].max())] if emitted.any() else None,
                  crossing_counts={str(int(i)): int(np.count_nonzero(crossing == i)) for i in np.unique(crossing[emitted])},
                  intensity_units="bolometric radiance W m^-2 sr^-1 (toy blackbody); display RGB illustrative",
                  solver_note="dense geometry may extend past first hit for reuse; radiance and constraint diagnostics stop at first opaque hit")
    (output/"diagnostics.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    return report


def repaint_surface(source, output, background):
    """Change background with identical opaque hits, shifts and intensities."""
    from .display import DiskDisplay
    source, output = Path(source), Path(output)
    report = json.loads((source/"diagnostics.json").read_text(encoding="utf-8"))
    with np.load(source/"disk_rays.npz") as data:
        codes, directions = data["status"], data["directions"]
        rgb = np.zeros(codes.shape+(3,))
        emitted, escaped = codes == 3, codes == 1
        rgb[emitted] = DiskDisplay(**report["display"]).linear_rgb(data["bolometric_intensity"][emitted], data["observed_temperature"][emitted])
        rgb[escaped] = srgb_to_linear(background.color(directions[escaped]))
        rgb[codes == 2] = (1, 0, 0.7)
    n = report["samples_per_axis"]
    h, w = codes.shape[0]//n, codes.shape[1]//n
    output.mkdir(parents=True, exist_ok=True)
    save_image(output/"shadow.png", linear_to_srgb(rgb.reshape(h, n, w, n, 3).mean(axis=(1, 3))))
    report.update(background=background.description, ray_data_source=str(source.resolve()))
    (output/"diagnostics.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
