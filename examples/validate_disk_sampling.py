"""Compare integrated disk observables at 32,64,192-pixel angular sampling."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import tomllib

import numpy as np

from numerical_observatory.background import CelestialGrid
from numerical_observatory.cli import load_configuration
from numerical_observatory.disk import make_disk
from numerical_observatory.surface_render import SurfacePathCache, render_surface


def measurements(path):
    with np.load(path/"disk_rays.npz") as data:
        emitted = data["status"] == 3
        intensity = data["bolometric_intensity"]
        crossing = data["crossing_index"]
        # Pinhole solid-angle element is proportional to (1+x²+y²)^(-3/2).
        h, w = emitted.shape
        yy, xx = np.mgrid[:h, :w]
        scale = np.tan(np.pi/6)/w
        x, y = (2*xx+1-w)*scale, (h-2*yy-1)*scale
        omega = (2*scale)**2/(1+x*x+y*y)**1.5
        return dict(width=w, height=h, emitting_solid_angle=float(omega[emitted].sum()),
                    bolometric_flux=float((intensity*omega).sum()),
                    secondary_solid_angle=float(omega[crossing >= 2].sum()),
                    approaching_flux=float((intensity*omega)[:, :w//2].sum()),
                    receding_flux=float((intensity*omega)[:, w//2:].sum()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("examples/disks"))
    parser.add_argument("--measure-only", action="store_true")
    args = parser.parse_args()
    metric, camera, settings = load_configuration("examples/disk_inclined.toml")
    config = tomllib.loads(Path("examples/disk_inclined.toml").read_text(encoding="utf-8"))
    if not args.measure_only:
        for width in (32, 64):
            current = replace(camera, width=width, height=width*3//4)
            cache = SurfacePathCache(metric, current, settings)
            for name, inclination in (("inclined", 60), ("edge_on", 85)):
                scene = make_disk(metric, dict(config["disk"], inclination_degrees=inclination))
                render_surface(metric, current, settings, args.output/f"sampling/{width}/{name}", scene,
                               background=CelestialGrid(), path_cache=cache)
            print(f"Completed {width}-pixel sampling", flush=True)
        # Same 32x24 scene, tighter ODE settings. This isolates integration
        # convergence from changing pixel positions or emitter color maps.
        current = replace(camera, width=32, height=24)
        tight = replace(settings, rtol=1e-11, atol=1e-13, max_step=2)
        cache = SurfacePathCache(metric, current, tight)
        for name, inclination in (("inclined", 60), ("edge_on", 85)):
            scene = make_disk(metric, dict(config["disk"], inclination_degrees=inclination))
            render_surface(metric, current, tight, args.output/f"sampling/tight/{name}", scene,
                           background=CelestialGrid(), path_cache=cache)
    report = {}
    for name in ("inclined", "edge_on"):
        records = [measurements(args.output/f"sampling/{width}/{name}") for width in (32, 64)]
        records.append(measurements(args.output/f"{name}_starfield"))
        reference = records[-1]
        for row in records:
            row["relative_flux_difference_vs_192"] = abs(row["bolometric_flux"]/reference["bolometric_flux"]-1)
        with np.load(args.output/f"sampling/32/{name}/disk_rays.npz") as a, np.load(args.output/f"sampling/tight/{name}/disk_rays.npz") as b:
            hits = (a["status"] == 3) & (b["status"] == 3)
            convergence = dict(classification_changes=int(np.count_nonzero(a["status"] != b["status"])),
                               crossing_order_changes=int(np.count_nonzero(a["crossing_index"] != b["crossing_index"])),
                               max_g_difference=float(np.max(np.abs(a["g"][hits]-b["g"][hits]))),
                               max_relative_intensity_difference=float(np.max(np.abs(a["bolometric_intensity"][hits]/b["bolometric_intensity"][hits]-1))))
        report[name] = dict(sampling=records, integration_convergence=convergence)
        # Acceptance checks on physical observables, independent of RGB display.
        assert convergence["classification_changes"] == 0
        assert convergence["crossing_order_changes"] == 0
        assert convergence["max_g_difference"] < 1e-6
        assert convergence["max_relative_intensity_difference"] < 1e-5
        assert records[1]["relative_flux_difference_vs_192"] < 0.05
        assert records[1]["relative_flux_difference_vs_192"] < records[0]["relative_flux_difference_vs_192"]
        assert abs(records[1]["emitting_solid_angle"]/reference["emitting_solid_angle"]-1) < 0.05
        assert reference["secondary_solid_angle"] > 0
        assert reference["approaching_flux"] > reference["receding_flux"]
    (args.output/"sampling_validation.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
