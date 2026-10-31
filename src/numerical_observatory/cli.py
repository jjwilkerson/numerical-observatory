"""Small TOML configuration and command-line interface."""
import argparse
from dataclasses import asdict
from pathlib import Path
import tomllib
import time

from .camera import Camera
from .background import make_background
from .geodesic import Integration
from .metric import Schwarzschild
from .render import render
from .disk import make_disk


def load_configuration(path=None):
    data = {} if path is None else tomllib.loads(Path(path).read_text(encoding="utf-8"))
    unknown = set(data) - {"mass", "camera", "integration", "background", "disk", "spacetime"}
    if unknown:
        raise ValueError(f"unknown configuration keys: {sorted(unknown)}")
    spacetime = dict(data.get("spacetime", {}))
    kind = spacetime.pop("type", "schwarzschild")
    if kind == "kerr":
        from .kerr import Kerr
        metric = Kerr(data.get("mass", 1.0), **spacetime)
    elif kind == "schwarzschild" and not spacetime:
        metric = Schwarzschild(data.get("mass", 1.0))
    else:
        raise ValueError("spacetime must select schwarzschild or kerr (with spin)")
    # Configuration lengths/times are in M, so mass changes preserve the scene.
    camera_options = dict(data.get("camera", {}))
    if kind == "kerr":
        from .kerr_support import inclined_camera_options
        camera_options = inclined_camera_options(metric, camera_options)
    camera_data = asdict(Camera()) | camera_options
    for key in ("position", "target"):
        camera_data[key] = tuple(metric.mass * v for v in camera_data[key])
    camera_data["time"] *= metric.mass
    integration_data = asdict(Integration()) | data.get("integration", {})
    for key in ("capture_radius", "escape_radius", "max_affine", "max_step", "kerr_source_radius"):
        integration_data[key] *= metric.mass
    return metric, Camera(**camera_data), Integration(**integration_data)


def main():
    parser = argparse.ArgumentParser(description="CPU Schwarzschild/Kerr observatory (luminous disks: Schwarzschild only)")
    parser.add_argument("--config", type=Path, help="TOML configuration (lengths in M)")
    parser.add_argument("--output", type=Path, default=Path("output"))
    parser.add_argument("--width", type=int, help="override pixel width")
    parser.add_argument("--height", type=int, help="override pixel height")
    parser.add_argument("--background", choices=("starfield", "grid"), help="override background mode")
    parser.add_argument("--method", choices=("ode", "quadrature"), default="ode")
    parser.add_argument("--samples", type=int, default=1, help="subpixel samples per image axis")
    parser.add_argument("--workers", type=int, default=1,
                        help="CPU processes: 1 serial (default), 0 automatic physical-core selection")
    parser.add_argument("--disk", action=argparse.BooleanOptionalAction, default=None,
                        help="override the optional disk enabled state (--no-disk disables it)")
    args = parser.parse_args()
    try:
        metric, camera, integration = load_configuration(args.config)
        camera_data = asdict(camera)
        for key in ("width", "height"):
            if getattr(args, key) is not None:
                camera_data[key] = getattr(args, key)
        camera = Camera(**camera_data)
        started, last_update = time.perf_counter(), [0.]
        def progress(row, total):
            elapsed = time.perf_counter()-started
            if row == total or elapsed-last_update[0] >= 5:
                estimate = elapsed*(total-row)/row if row else float("inf")
                print(f"Completed rows {row}/{total}; elapsed {elapsed:.1f}s; estimated remaining {estimate:.1f}s", flush=True)
                last_update[0] = elapsed
        def work_progress(done, total, pixels, total_pixels):
            elapsed = time.perf_counter()-started
            if elapsed-last_update[0] >= 5:
                estimate = elapsed*(total_pixels-pixels)/pixels if pixels else float("inf")
                print(f"Completed batches {done}/{total}, work units {pixels}/{total_pixels}; elapsed {elapsed:.1f}s; estimated remaining {estimate:.1f}s", flush=True)
                last_update[0] = elapsed
        progress.work = work_progress
        data = {} if args.config is None else tomllib.loads(args.config.read_text(encoding="utf-8"))
        background_options = data.get("background", {})
        if args.background:
            background_options["mode"] = args.background
        background = make_background(background_options)
        disk_options = dict(data.get("disk", {}))
        if args.disk is not None:
            disk_options["enabled"] = args.disk
        disk = make_disk(metric, disk_options)
        diagnostics = render(metric, camera, integration, args.output, progress, background,
                             method=args.method, samples=args.samples, disk=disk, workers=args.workers)
    except KeyboardInterrupt:
        parser.exit(130, "Render interrupted; workers stopped; no incomplete render published.\n")
    except (ValueError, TypeError, OSError, RuntimeError) as exc:
        parser.error(str(exc))
    print(f"Saved {args.output / 'shadow.png'} and diagnostics.json")
    print(diagnostics["counts"])


if __name__ == "__main__":
    main()
