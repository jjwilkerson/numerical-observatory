"""Serial CPU rendering, portable image output, and compact JSON diagnostics."""
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import scipy

from . import __version__
from .background import Starfield
from .symmetry import CenteredRayCache


def render(metric, camera, integration, output, progress=None, background=None,
           use_symmetry=True):
    # Validate the camera before allocating or tracing an image.
    camera.tetrad(metric)
    image = np.zeros((camera.height, camera.width, 3))
    background = Starfield() if background is None else background
    tracer = CenteredRayCache(metric, camera, integration, enabled=use_symmetry)
    directions = np.full((camera.height, camera.width, 3), np.nan)
    statuses = np.zeros((camera.height, camera.width), dtype=np.uint8)
    counts, reasons = Counter(), Counter()
    maxima = dict(null_error=0.0, energy_drift=0.0, angular_momentum_drift=0.0)
    start = time.perf_counter()
    for row, col, momentum in camera.rays(metric):
        result = tracer.ray(row, col, momentum)
        counts[result.status] += 1
        reasons[result.reason] += 1
        if result.status == "escaped":
            directions[row, col] = result.direction
            statuses[row, col] = 1
            image[row, col] = background.color(result.direction)
        elif result.status == "unresolved":
            statuses[row, col] = 2
            image[row, col] = (1.0, 0.0, 0.7)
        maxima["null_error"] = max(maxima["null_error"], result.max_null_error)
        maxima["energy_drift"] = max(maxima["energy_drift"], result.energy_drift)
        maxima["angular_momentum_drift"] = max(maxima["angular_momentum_drift"], result.angular_momentum_drift)
        if progress and col == camera.width - 1:
            progress(row + 1, camera.height)
    radius = np.linalg.norm(camera.position)
    diagnostics = {
        "version": __version__, "numpy": np.__version__, "scipy": scipy.__version__,
        "mass": metric.mass, "camera": asdict(camera), "integration": asdict(integration),
        "counts": {k: counts[k] for k in ("captured", "escaped", "unresolved")},
        "termination_reasons": dict(reasons), "maxima": maxima,
        "diagnostic_sampling": "accepted steps and dense-output midpoints; null error = |g^ab p_a p_b| / p_t(initial)^2",
        "total_steps": tracer.steps, "rhs_evaluations": tracer.evaluations,
        "integrated_rays": tracer.integrated_rays,
        "rotational_symmetry_reuse": tracer.enabled,
        "elapsed_seconds": time.perf_counter() - start,
        "analytic_critical_impact_parameter": metric.critical_impact_parameter,
        "analytic_shadow_radius_degrees": float(np.rad2deg(metric.shadow_angle(radius))) if radius > 3*metric.mass else None,
        "background": background.description,
    }
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.uint8(np.round(np.clip(image, 0, 1) * 255))).save(output / "shadow.png")
    # Save computed directions for later background/exposure changes without ODEs.
    # Codes: 0=captured, 1=escaped, 2=unresolved. Non-escaping directions are NaN.
    np.savez_compressed(output / "rays.npz", directions=directions, status=statuses)
    (output / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return diagnostics
