"""Render a centered Schwarzschild composition using exact orbital quadrature.

The ordinary numerical-observatory CLI also accepts this configuration and
uses the full ODE tracer. This example uses the independent orbit quadrature
already used in validation, and checks representative rays against that ODE.
No interpolation of ray directions, image warping, or artificial lens mask.
"""
import argparse
import json
from pathlib import Path
import time
import tomllib

import numpy as np
from PIL import Image

from numerical_observatory.background import Starfield
from numerical_observatory.cli import load_configuration
from numerical_observatory.geodesic import trace
from numerical_observatory.validation import orbit_direction


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("examples/composition.toml"))
    parser.add_argument("--output", type=Path, default=Path("examples/composition"))
    args = parser.parse_args()
    config = tomllib.loads(args.config.read_text(encoding="utf-8"))
    metric, camera, integration = load_configuration(args.config)
    background = Starfield(**config.get("background", {}))
    forward, right, up = camera.axes()
    radius = np.linalg.norm(camera.position)
    radial = np.asarray(camera.position)/radius
    if radius <= 3*metric.mass or np.linalg.norm(forward+radial) > 1e-14:
        raise ValueError("This example requires a centered camera outside r=3M")

    start = time.perf_counter()
    h, w = camera.height, camera.width
    yy, xx = np.mgrid[:h, :w]
    ix, iy = 2*xx+1-w, h-2*yy-1
    squared_radius = ix*ix + iy*iy
    scale = np.tan(np.deg2rad(camera.fov_degrees)/2)/w
    keys, reverse = np.unique(squared_radius, return_inverse=True)
    angles = np.arctan(np.sqrt(keys)*scale)
    critical = metric.shadow_angle(radius)
    escaped = angles > critical
    orbit = np.full((len(keys), 3), np.nan)
    for i in np.flatnonzero(escaped):
        orbit[i] = orbit_direction(angles[i], radius/metric.mass)
    orbit = orbit[reverse].reshape(h, w, 3)
    escaped_pixels = escaped[reverse].reshape(h, w)
    transverse = (ix[..., None]*right + iy[..., None]*up)
    transverse /= np.maximum(np.sqrt(squared_radius)[..., None], 1)
    directions = orbit[..., 0, None]*transverse + orbit[..., 2, None]*radial
    image = np.zeros((h, w, 3))
    image[escaped_pixels] = background.color(directions[escaped_pixels])

    # Cross-check both sides of the shadow and the lensing/outer field using
    # independent Hamiltonian ODE integration, including a nonzero azimuth.
    checks = []
    test_angles = [critical*0.8, critical*(1-1e-4), critical*(1+1e-4),
                   critical*1.1, critical*2, np.sqrt(4*metric.mass/radius),
                   float(angles.max())*0.6, float(angles.max())]
    transverse_check = (right+2*up)/np.sqrt(5)
    for angle in test_angles:
        initial = np.cos(angle)*forward + np.sin(angle)*transverse_check
        result = trace(metric, camera.event, camera.momentum(metric, initial), integration)
        expected = "captured" if angle < critical else "escaped"
        if result.status != expected:
            raise RuntimeError(f"ODE classification disagreement at angle {angle}: {result.reason}")
        error = None
        if expected == "escaped":
            direction = orbit_direction(angle, radius/metric.mass)
            reference = direction[0]*transverse_check + direction[2]*radial
            error = float(np.linalg.norm(reference-result.direction))
            if error > 2e-5:
                raise RuntimeError(f"ODE direction disagreement: {error}")
        checks.append(dict(angle_radians=float(angle), status=result.status,
                           direction_error=error, ode_max_null_error=result.max_null_error))

    # An unlensed view of exactly the same sky patch makes the composition and
    # actual geodesic distortion easy to compare; it is not a second disk model.
    straight = forward + scale*(ix[..., None]*right + iy[..., None]*up)
    straight /= np.linalg.norm(straight, axis=-1, keepdims=True)
    unlensed = background.color(straight)
    args.output.mkdir(parents=True, exist_ok=True)
    for name, rgb in (("shadow.png", image), ("unlensed.png", unlensed)):
        Image.fromarray(np.uint8(np.round(np.clip(rgb, 0, 1)*255))).save(args.output/name)
    np.savez_compressed(args.output/"rays.npz", directions=directions,
                        status=escaped_pixels.astype(np.uint8))
    report = dict(method="Schwarzschild exact orbit quadrature, no direction interpolation",
                  config=config, background=background.description,
                  counts=dict(captured=int(np.count_nonzero(~escaped_pixels)),
                              escaped=int(np.count_nonzero(escaped_pixels)), unresolved=0),
                  distinct_orbits=len(keys),
                  analytic_shadow_radius_pixels=float(np.tan(critical)/(2*scale)),
                  ode_cross_checks=checks,
                  diagnostic_scope="Null residuals apply only to the eight ODE cross-check rays; the image uses orbital quadrature.",
                  elapsed_seconds=time.perf_counter()-start)
    (args.output/"diagnostics.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
