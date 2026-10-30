"""Small TOML configuration and command-line interface."""
import argparse
from dataclasses import asdict
from pathlib import Path
import tomllib

from .camera import Camera
from .background import make_background
from .geodesic import Integration
from .metric import Schwarzschild
from .render import render


def load_configuration(path=None):
    data = {} if path is None else tomllib.loads(Path(path).read_text(encoding="utf-8"))
    unknown = set(data) - {"mass", "camera", "integration", "background"}
    if unknown:
        raise ValueError(f"unknown configuration keys: {sorted(unknown)}")
    metric = Schwarzschild(data.get("mass", 1.0))
    # Configuration lengths/times are in M, so mass changes preserve the scene.
    camera_data = asdict(Camera()) | data.get("camera", {})
    for key in ("position", "target"):
        camera_data[key] = tuple(metric.mass * v for v in camera_data[key])
    camera_data["time"] *= metric.mass
    integration_data = asdict(Integration()) | data.get("integration", {})
    for key in ("capture_radius", "escape_radius", "max_affine", "max_step"):
        integration_data[key] *= metric.mass
    return metric, Camera(**camera_data), Integration(**integration_data)


def main():
    parser = argparse.ArgumentParser(description="CPU Schwarzschild black-hole shadow tracer")
    parser.add_argument("--config", type=Path, help="TOML configuration (lengths in M)")
    parser.add_argument("--output", type=Path, default=Path("output"))
    parser.add_argument("--width", type=int, help="override pixel width")
    parser.add_argument("--height", type=int, help="override pixel height")
    parser.add_argument("--background", choices=("starfield", "grid"), help="override background mode")
    parser.add_argument("--method", choices=("ode", "quadrature"), default="ode")
    parser.add_argument("--samples", type=int, default=1, help="subpixel samples per axis for quadrature rendering")
    args = parser.parse_args()
    try:
        metric, camera, integration = load_configuration(args.config)
        camera_data = asdict(camera)
        for key in ("width", "height"):
            if getattr(args, key) is not None:
                camera_data[key] = getattr(args, key)
        camera = Camera(**camera_data)
        def progress(row, total):
            if row % max(1, total // 8) == 0 or row == total:
                print(f"Rows {row}/{total}", flush=True)
        data = {} if args.config is None else tomllib.loads(args.config.read_text(encoding="utf-8"))
        background_options = data.get("background", {})
        if args.background:
            background_options["mode"] = args.background
        background = make_background(background_options)
        diagnostics = render(metric, camera, integration, args.output, progress, background,
                             method=args.method, samples=args.samples)
    except (ValueError, TypeError, OSError) as exc:
        parser.error(str(exc))
    print(f"Saved {args.output / 'shadow.png'} and diagnostics.json")
    print(diagnostics["counts"])


if __name__ == "__main__":
    main()
