"""CPU rendering with optional spawn workers, portable output and diagnostics."""
from collections import Counter
from dataclasses import asdict, replace
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import scipy

from . import __version__
from .background import Starfield
from .symmetry import CenteredRayCache
from .metric import Schwarzschild
from .schwarzschild_orbit import compatible


def render(metric, camera, integration, output, progress=None, background=None,
           use_symmetry=True, domain=None, method="ode", samples=1, disk=None, workers=1):
    from .parallel import resolve_workers
    from .output import staged_render
    selected = resolve_workers(workers)
    return staged_render(output, lambda stage: _render(metric, camera, integration, stage, progress,
                         background, use_symmetry, domain, method, samples, disk, selected))


def _render(metric, camera, integration, output, progress=None, background=None,
            use_symmetry=True, domain=None, method="ode", samples=1, disk=None, workers=1):
    if method not in ("ode", "quadrature"):
        raise ValueError("method must be ode or quadrature")
    if not isinstance(samples, int) or samples < 1:
        raise ValueError("samples must be a positive integer")
    if disk is not None:
        from .kerr import Kerr
        if type(metric) is Kerr:
            from .surfaces import KerrEquatorialAnnulus
            from .emitter_motion import KerrCircular
            if (not isinstance(disk.surface, KerrEquatorialAnnulus) or not isinstance(disk.motion, KerrCircular)
                    or disk.surface.metric != metric or disk.motion.metric != metric):
                raise ValueError("Kerr rendering requires a matching equatorial Kerr disk, not Schwarzschild disk physics")
        from .surface_render import render_surface
        return render_surface(metric, camera, integration, output, disk, background, samples,
                              domain, progress, requested_method=method, use_symmetry=use_symmetry, workers=workers)
    if method == "quadrature" and compatible(metric, camera) and domain is None:
        from .diagnostic_render import render_quadrature
        return render_quadrature(metric, camera, integration, output,
                                 Starfield() if background is None else background, samples, workers=workers,
                                 progress=progress)
    if samples > 1:
        from .diagnostic_render import shade, save_image
        background = Starfield() if background is None else background
        sampled_camera = replace(camera, width=camera.width*samples, height=camera.height*samples)
        report = _render(metric, sampled_camera, integration, output, progress, background,
                         use_symmetry=use_symmetry, domain=domain, method="ode", workers=workers)
        output = Path(output)
        with np.load(output/"rays.npz") as rays:
            directions, status = rays["directions"], rays["status"]
            extra = {key: rays[key] for key in rays.files if key not in ("directions", "status", "samples")}
        save_image(output/"shadow.png", shade(directions, status, background, samples))
        np.savez_compressed(output/"rays.npz", directions=directions, status=status, samples=samples, **extra)
        report.update(camera=asdict(camera), samples_per_axis=samples, requested_method=method,
                      count_units="subpixel rays")
        (output/"diagnostics.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
        return report
    # Validate the camera before allocating or tracing an image.
    camera.tetrad(metric)
    image = np.zeros((camera.height, camera.width, 3))
    background = Starfield() if background is None else background
    tracer = CenteredRayCache(metric, camera, integration, enabled=use_symmetry, domain=domain)
    if workers > 1:
        from .parallel import ParallelRays
        tracer = ParallelRays(metric, camera, integration, workers, use_symmetry, domain)
    directions = np.full((camera.height, camera.width, 3), np.nan)
    statuses = np.zeros((camera.height, camera.width), dtype=np.uint8)
    from .kerr import Kerr
    is_kerr = type(metric) is Kerr
    conservation = ({key: np.full((camera.height, camera.width), np.nan)
                     for key in ("null_error", "energy_drift", "axial_momentum_drift", "carter_drift")}
                    if is_kerr else {})
    constants = np.full((camera.height, camera.width, 3), np.nan) if is_kerr else None
    counts, reasons = Counter(), Counter()
    maxima = dict(null_error=0.0, energy_drift=0.0, angular_momentum_drift=0.0)
    start = time.perf_counter()
    remaining = np.full(camera.height, camera.width)
    completed = 0
    results = (tracer.results(progress) if workers > 1 else
               ((row, col, momentum, tracer.ray(row, col, momentum))
                for row, col, momentum in camera.rays(metric)))
    from contextlib import closing
    with closing(results):
        for row, col, momentum, result in results:
            counts[result.status] += 1
            reasons[result.reason] += 1
            if is_kerr:
                for key, value in (("null_error", result.max_null_error), ("energy_drift", result.energy_drift),
                                   ("axial_momentum_drift", result.angular_momentum_drift),
                                   ("carter_drift", result.carter_drift)):
                    conservation[key][row, col] = value
                constants[row, col] = result.kerr_constants
            if result.status == "escaped":
                directions[row, col] = result.direction
                statuses[row, col] = 1
                image[row, col] = background.color(result.direction)
            elif result.status == "unresolved":
                statuses[row, col] = 2
                image[row, col] = (1.0, 0.0, 0.7)
            maxima["null_error"] = max(maxima["null_error"], result.max_null_error)
            for key in ("energy_drift", "angular_momentum_drift"):
                value = getattr(result, key)
                maxima[key] = max(maxima[key], value) if value is not None else None
            remaining[row] -= 1
            if remaining[row] == 0:
                completed += 1
                if progress:
                    progress(completed, camera.height)
    radius = np.linalg.norm(camera.position)
    diagnostics = {
        "method": "ode", "requested_method": method,
        "version": __version__, "numpy": np.__version__, "scipy": scipy.__version__,
        "mass": getattr(metric, "mass", None), "camera": asdict(camera), "integration": asdict(integration),
        "counts": {k: counts[k] for k in ("captured", "escaped", "unresolved")},
        "termination_reasons": dict(reasons), "maxima": maxima,
        "diagnostic_sampling": "accepted steps and dense-output midpoints; null error = |g^ab p_a p_b| / p_t(initial)^2",
        "total_steps": tracer.steps, "rhs_evaluations": tracer.evaluations,
        "integrated_rays": tracer.integrated_rays,
        "rotational_symmetry_reuse": tracer.enabled,
        "elapsed_seconds": time.perf_counter() - start,
        "analytic_critical_impact_parameter": metric.critical_impact_parameter if type(metric) is Schwarzschild else None,
        "analytic_shadow_radius_degrees": float(np.rad2deg(metric.shadow_angle(radius))) if type(metric) is Schwarzschild and radius > 3*metric.mass else None,
        "background": background.description,
    }
    if workers > 1:
        diagnostics["execution"] = tracer.statistics
    if is_kerr:
        diagnostics.update(spacetime={"type": "kerr", "spin": metric.spin, "a": metric.a,
                                      "axis": "+Z", "coordinates": "outgoing Cartesian Kerr-Schild"},
                           horizon_radius=metric.horizon_radius,
                           capture_radial_coordinate=integration.kerr_capture_fraction*metric.horizon_radius,
                           background_direction="Hamiltonian continuation to kerr_source_radius, normalized outgoing Cartesian tangent; residual bending O(M*b/R^2)",
                           conservation_normalization="|delta E|/|E0|, |delta Lz|/(M|E0|), |delta Q|/(M E0)^2; signed past-ray E,Lz,Q saved",
                           optimization_note="Kerr uses general ODEs; Schwarzschild quadrature and spherical rotational reuse disabled")
        diagnostics["maxima"].update(axial_momentum_drift=float(np.max(conservation["axial_momentum_drift"])),
                                      carter_drift=float(np.max(conservation["carter_drift"])))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.uint8(np.round(np.clip(image, 0, 1) * 255))).save(output / "shadow.png")
    # Save computed directions for later background/exposure changes without ODEs.
    # Codes: 0=captured, 1=escaped, 2=unresolved. Non-escaping directions are NaN.
    np.savez_compressed(output / "rays.npz", directions=directions, status=statuses,
                        **conservation, **({"kerr_constants": constants} if is_kerr else {}))
    (output / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return diagnostics
