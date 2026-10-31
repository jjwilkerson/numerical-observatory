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
        self.integrated_rays = self.steps = self.evaluations = 0

    def _trace(self, momentum):
        sol = trace(self.metric, self.camera.event, momentum, self.integration,
                    domain=self.domain, return_solution=True)[1]
        self.integrated_rays += 1
        self.steps += len(sol.t)-1
        self.evaluations += sol.nfev
        return sol

    def solution(self, row, col, momentum):
        camera = self.camera
        ix, iy = 2*col+1-camera.width, camera.height-2*row-1
        key = ix*ix+iy*iy
        if not self.enabled:
            return self._trace(momentum)
        if key not in self.paths:
            offset = np.sqrt(key)*np.tan(np.deg2rad(camera.fov_degrees)/2)/camera.width
            canonical = camera.momentum(self.metric, self.forward+offset*self.right)
            self.paths[key] = self._trace(canonical)
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

    def ray(self, row, col, momentum, surface):
        from .kerr import Kerr
        if type(self.metric) is Kerr:
            # No spherical reuse. Passing the surface into the general tracer
            # keeps the Kerr domain and audits only the physical opaque prefix.
            result = trace(self.metric, self.camera.event, momentum, self.integration,
                           domain=self.domain, surface=surface)
            self.integrated_rays += 1
            self.steps += result.steps
            self.evaluations += result.evaluations
            return result
        sol = self.solution(row, col, momentum)
        return summarize_solution(self.metric, self.camera.event, momentum, self.integration, sol,
                                  domain=self.domain, builtin_domain=self.domain is None, surface=surface)


def render_surface(metric, camera, integration, output, scene, background=None,
                   samples=1, domain=None, progress=None, path_cache=None, requested_method="ode",
                   use_symmetry=True, workers=1):
    from .parallel import resolve_workers
    workers = resolve_workers(workers)
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
    if workers > 1:
        if path_cache is not None:
            raise ValueError("parallel rendering cannot accept an external dense path cache; use workers=1")
        from .parallel import ParallelRays
        cache = ParallelRays(metric, sampled, integration, workers, use_symmetry, domain, scene.surface)
    initial_work = (cache.integrated_rays, cache.steps, cache.evaluations)
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
    from .kerr import Kerr
    is_kerr = type(metric) is Kerr
    from .backend_records import BackendRecords
    backend_records=BackendRecords((h,w),integration) if is_kerr else None
    extra = ({key: np.full((h,w)+tail, np.nan) for key,tail in (
                ("hit_event",(4,)), ("hit_momentum",(4,)), ("hit_affine",()),
                ("emission_azimuth_degrees",()), ("emitted_bolometric_intensity",()),
                ("emitted_temperature",()), ("null_error",()), ("energy_drift",()),
                ("axial_momentum_drift",()), ("carter_drift",()), ("kerr_constants",(3,)))}
             if is_kerr else {})
    null_max = 0.0
    counts, reasons = Counter(), Counter()
    start = time.perf_counter()
    def serial_results():
        for row, col, momentum in sampled.rays(metric):
            result = cache.ray(row, col, momentum, scene.surface)
            yield row, col, momentum, result
    results = cache.results(progress) if workers > 1 else serial_results()
    remaining, completed = np.full(h, w), 0
    from contextlib import closing
    with closing(results):
        for row, col, momentum, result in results:
            counts[result.status] += 1
            reasons[result.reason] += 1
            null_max = max(null_max, result.max_null_error)
            if is_kerr:
                backend_records.record(row,col,result)
                for key,value in (("null_error",result.max_null_error), ("energy_drift",result.energy_drift),
                                  ("axial_momentum_drift",result.angular_momentum_drift),
                                  ("carter_drift",result.carter_drift), ("kerr_constants",result.kerr_constants)):
                    extra[key][row,col] = value
            if result.status == "emitted":
                hit = result.hit
                g = frequency_shift(momentum, observer, hit.momentum, scene.motion.four_velocity(hit.event))
                bolometric = scene.emission.observed_bolometric(hit.event, g)
                observed_temperature = g*scene.emission.temperature(hit.event)
                codes[row, col] = 3
                shift[row, col], intensity[row, col] = g, bolometric
                temperature[row, col] = observed_temperature
                hit_radius[row, col] = (scene.surface.radius(hit.event) if hasattr(scene.surface, "radius")
                                       else np.linalg.norm(hit.event[1:4]))
                hit_position[row, col] = hit.event[1:4]
                crossing[row, col] = hit.crossing_index
                rgb[row, col] = scene.display.linear_rgb(bolometric, observed_temperature)
                if is_kerr:
                    extra["hit_event"][row,col] = hit.event
                    extra["hit_momentum"][row,col] = hit.momentum
                    extra["hit_affine"][row,col] = hit.affine
                    extra["emission_azimuth_degrees"][row,col] = np.degrees(np.arctan2(hit.event[2],hit.event[1])) % 360
                    extra["emitted_temperature"][row,col] = scene.emission.temperature(hit.event)
                    extra["emitted_bolometric_intensity"][row,col] = scene.emission.bolometric(hit.event)
            elif result.status == "escaped":
                codes[row, col] = 1
                directions[row, col] = result.direction
                rgb[row, col] = srgb_to_linear(background.color(result.direction))
            elif result.status == "unresolved":
                codes[row, col] = 2
                rgb[row, col] = (1, 0, 0.7)
            remaining[row] -= 1
            if remaining[row] == 0:
                completed += 1
                if progress:
                    progress(completed, h)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    image = linear_to_srgb(rgb.reshape(camera.height, samples, camera.width, samples, 3).mean(axis=(1, 3)))
    save_image(output/"shadow.png", image)
    np.savez_compressed(output/"disk_rays.npz", status=codes, g=shift, bolometric_intensity=intensity,
                        observed_temperature=temperature, hit_radius=hit_radius, hit_position=hit_position,
                        crossing_index=crossing, directions=directions, samples=samples, **extra,
                        **(backend_records.arrays() if is_kerr else {}))
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
    report.update(integrated_rays=cache.integrated_rays-initial_work[0], total_steps=cache.steps-initial_work[1],
                  rhs_evaluations=cache.evaluations-initial_work[2])
    if workers > 1:
        report["execution"] = cache.statistics
    if is_kerr:
        report.update(backend_records.metadata())
        from .emitter_motion import kerr_isco
        report.update(spacetime={"type":"kerr", "spin":metric.spin, "a":metric.a,
                                 "axis":"+Z", "coordinates":"outgoing Cartesian Kerr-Schild"},
                      orbit=scene.motion.orbit, isco_radius=kerr_isco(metric,scene.motion.orbit),
                      resolved_inner_radius=scene.surface.inner_radius,
                      radius_convention="Kerr oblate r, not Euclidean or cylindrical radius",
                      azimuth_convention="outgoing Cartesian KS azimuth atan2(y,x), [0,360) degrees; zero +X, positive toward +Y about +Z; not BL azimuth",
                      horizon_radius=metric.horizon_radius,
                      capture_radial_coordinate=integration.kerr_capture_fraction*metric.horizon_radius,
                      background_direction="same Hamiltonian continuation and far-field tangent as disk-disabled Kerr",
                      emission_model="phenomenological power-law temperature vs Kerr r; not Novikov-Thorne or an equal-accretion-rate comparison",
                      conservation_normalization="null: E0^2; delta E: |E0|; delta Lz: M|E0|; delta Q: (M E0)^2; samples stop at first opaque hit; escaping rays include tail",
                      conservation_maxima={key:float(np.max(extra[key])) for key in ("null_error","energy_drift","axial_momentum_drift","carter_drift")},
                      kerr_multilayer_supported=False)
        if integration.integrator == "carter":
            report["background_direction"]="Exact separated angular tail to the same finite Kerr source radius, final Cartesian tangent; fallback uses the unchanged Hamiltonian mapping"
    (output/"diagnostics.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    return report


def repaint_surface(source, output, background=None, display=None):
    """Change display/background only; physical numerical arrays are read-only."""
    from .display import DiskDisplay
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("repaint output must be separate from its source")
    report = json.loads((source/"diagnostics.json").read_text(encoding="utf-8"))
    if display:
        report["display"] = report["display"] | display
    transform = DiskDisplay(**report["display"])
    if background is None:
        from .background import make_background
        sky = report["background"]
        grid = sky["model"] == "celestial coordinate grid"
        keys = ("longitude_degrees","latitude_degrees","roll_degrees","spacing_degrees","line_width_degrees") if grid else (
            "longitude_degrees","latitude_degrees","roll_degrees","exposure","resolution")
        background = make_background(dict(mode="grid" if grid else "starfield",**{k:sky[k] for k in keys if k in sky}))
    ray_source = source/"disk_rays.npz"
    if not ray_source.exists() and report.get("ray_data_source"):
        ray_source = Path(report["ray_data_source"])
        if not ray_source.is_absolute():
            ray_source = source/ray_source
        if ray_source.suffix != ".npz":
            ray_source = ray_source/"disk_rays.npz"
    with np.load(ray_source) as data:
        codes, directions = data["status"], data["directions"]
        rgb = np.zeros(codes.shape+(3,))
        emitted, escaped = codes == 3, codes == 1
        rgb[emitted] = transform.linear_rgb(data["bolometric_intensity"][emitted], data["observed_temperature"][emitted])
        rgb[escaped] = srgb_to_linear(background.color(directions[escaped]))
        rgb[codes == 2] = (1, 0, 0.7)
    n = report["samples_per_axis"]
    h, w = codes.shape[0]//n, codes.shape[1]//n
    output.mkdir(parents=True, exist_ok=True)
    save_image(output/"shadow.png", linear_to_srgb(rgb.reshape(h, n, w, n, 3).mean(axis=(1, 3))))
    report.update(background=background.description, ray_data_source=str(ray_source.resolve()),
                  repaint_only=True, geodesics_retraced=False)
    (output/"diagnostics.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Recolor saved opaque disk rays without tracing")
    parser.add_argument("--input",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--palette",choices=("blackbody","thermal","gray"))
    parser.add_argument("--exposure",type=float)
    parser.add_argument("--tone-mapping",choices=("reinhard","linear"))
    parser.add_argument("--background",choices=("grid","starfield"))
    args=parser.parse_args()
    changes={key:getattr(args,key) for key in ("palette","exposure","tone_mapping") if getattr(args,key) is not None}
    from .background import make_background
    try:
        repaint_surface(args.input,args.output,None if args.background is None else make_background(dict(mode=args.background)),changes)
    except (ValueError, OSError, KeyError) as exc:
        parser.error(str(exc))
