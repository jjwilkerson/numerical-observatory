"""Reproduce disk / no-disk, starfield / grid pairs with shared geometry."""
import argparse
from dataclasses import replace
from pathlib import Path
import tomllib

from numerical_observatory.background import make_background
from numerical_observatory.cli import load_configuration
from numerical_observatory.disk import make_disk
from numerical_observatory.render import render
from numerical_observatory.surface_render import SurfacePathCache, render_surface, repaint_surface


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, default=192)
    parser.add_argument("--height", type=int, default=144)
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--output", type=Path, default=Path("examples/disks"))
    args = parser.parse_args()
    metric, camera, integration = load_configuration("examples/disk_inclined.toml")
    camera = replace(camera, width=args.width, height=args.height)
    config = tomllib.loads(Path("examples/disk_inclined.toml").read_text(encoding="utf-8"))
    sampled = replace(camera, width=camera.width*args.samples, height=camera.height*args.samples)
    cache = SurfacePathCache(metric, sampled, integration)
    for name, inclination in (("inclined", 60.0), ("edge_on", 85.0)):
        options = dict(config["disk"], inclination_degrees=inclination)
        scene = make_disk(metric, options)
        def progress(row, total):
            if row % max(1, total//12) == 0 or row == total:
                print(f"{name}: rows {row}/{total}", flush=True)
        report = render_surface(metric, camera, integration, args.output/f"{name}_starfield", scene,
                                make_background(config["background"]), samples=args.samples,
                                path_cache=cache, progress=progress)
        print(name, report["counts"], report["crossing_counts"], flush=True)
        repaint_surface(args.output/f"{name}_starfield", args.output/f"{name}_grid",
                        make_background(dict(config["background"], mode="grid")))
    for mode in ("starfield", "grid"):
        render(metric, camera, integration, args.output/f"no_disk_{mode}",
               background=make_background(dict(config["background"], mode=mode)),
               method="quadrature", samples=args.samples)


if __name__ == "__main__":
    main()
