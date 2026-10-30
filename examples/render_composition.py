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

from numerical_observatory.background import make_background
from numerical_observatory.schwarzschild_orbit import compatible, ray_map
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
    background = make_background(config.get("background", {}))
    forward, right, up = camera.axes()
    radius = np.linalg.norm(camera.position)
    radial = np.asarray(camera.position)/radius
    if not compatible(metric, camera):
        raise ValueError("This example requires a centered camera outside r=3M")

    start = time.perf_counter()
    h, w = camera.height, camera.width
    yy, xx = np.mgrid[:h, :w]
    ix, iy = 2*xx+1-w, h-2*yy-1
    scale = np.tan(np.deg2rad(camera.fov_degrees)/2)/w
    maximum_angle = np.arctan(np.hypot(ix, iy).max()*scale)
    critical = metric.shadow_angle(radius)
    directions, status, distinct = ray_map(metric, camera)
    escaped_pixels = status == 1
    image = np.zeros((h, w, 3))
    image[escaped_pixels] = background.color(directions[escaped_pixels])
    image[status == 2] = (1, 0, 0.7)

    # Cross-check both sides of the shadow and the lensing/outer field using
    # independent Hamiltonian ODE integration, including a nonzero azimuth.
    checks = []
    test_angles = [critical*0.8, critical*(1-1e-4), critical*(1+1e-4),
                   critical*1.1, critical*2, np.sqrt(4*metric.mass/radius),
                   float(maximum_angle)*0.6, float(maximum_angle)]
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
                        status=status)
    report = dict(method="Schwarzschild exact orbit quadrature, no direction interpolation",
                  config=config, background=background.description,
                  counts=dict(captured=int(np.count_nonzero(status == 0)),
                              escaped=int(np.count_nonzero(escaped_pixels)), unresolved=int(np.count_nonzero(status == 2))),
                  distinct_orbits=distinct,
                  analytic_shadow_radius_pixels=float(np.tan(critical)/(2*scale)),
                  ode_cross_checks=checks,
                  diagnostic_scope="Null residuals apply only to the eight ODE cross-check rays; the image uses orbital quadrature.",
                  elapsed_seconds=time.perf_counter()-start)
    (args.output/"diagnostics.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
