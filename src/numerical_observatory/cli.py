"""Small TOML configuration and command-line interface."""
import argparse
from dataclasses import asdict, replace
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
    unknown = set(data) - {"mass", "camera", "integration", "background", "disk", "spacetime", "geodesics", "numerical_domain"}
    if unknown:
        raise ValueError(f"unknown configuration keys: {sorted(unknown)}")
    spacetime = dict(data.get("spacetime", {}))
    kind = spacetime.pop("type", "schwarzschild")
    if kind == 'numerical':
        from .numerical_metric import NumericalMetric
        if data.get('mass',1.)!=1.:raise ValueError('numerical data retains stored code coordinates; implicit mass rescaling is forbidden')
        directory=Path(spacetime.pop('dataset'))
        if not directory.is_absolute():directory=(Path(path).resolve().parent if path else Path.cwd())/directory
        metric=NumericalMetric(directory,domain=data.get('numerical_domain'),**spacetime)
    elif kind == "kerr":
        from .kerr import Kerr
        metric = Kerr(data.get("mass", 1.0), **spacetime)
    elif kind == "schwarzschild" and not spacetime:
        metric = Schwarzschild(data.get("mass", 1.0))
    elif kind == "binary_ks_approx":
        from .binary_ks_approx import BinaryKSApprox
        scale=data.get('mass',1.)
        if not isinstance(scale,(int,float)) or not 0 < scale < float('inf'):
            raise ValueError('binary reference mass must be finite and positive')
        masses=spacetime.pop('masses',[.5,.5])
        centers=spacetime.pop('centers',[[-6,0,0],[6,0,0]])
        if spacetime: raise ValueError('binary spacetime accepts only masses and centers')
        metric=BinaryKSApprox(tuple(scale*v for v in masses),tuple(tuple(scale*v for v in c) for c in centers))
    elif kind == 'binary_ks_orbit_approx':
        from .binary_ks_orbit_approx import BinaryKSOrbitApprox
        from .trajectories import CircularOrbit
        scale=data.get('mass',1.)
        if not isinstance(scale,(int,float)) or not 0<scale<float('inf'):raise ValueError('reference mass must be finite and positive')
        masses=tuple(scale*v for v in spacetime.pop('masses',[.5,.5]))
        separation=scale*spacetime.pop('separation',30.)
        phase=spacetime.pop('phase_degrees',0.)
        epoch=scale*spacetime.pop('epoch',0.)
        center=tuple(scale*v for v in spacetime.pop('barycenter',[0.,0.,0.]))
        if spacetime:raise ValueError('orbit spacetime accepts masses, separation, phase_degrees, epoch, barycenter only')
        metric=BinaryKSOrbitApprox(masses,CircularOrbit(masses,separation,phase,epoch,center))
    else:
        raise ValueError("spacetime must select schwarzschild, kerr, binary_ks_approx, binary_ks_orbit_approx, or numerical")
    # Configuration lengths/times are in M, so mass changes preserve the scene.
    camera_options = dict(data.get("camera", {}))
    if kind == "kerr":
        from .kerr_support import inclined_camera_options
        camera_options = inclined_camera_options(metric, camera_options)
    camera_data = asdict(Camera()) | camera_options
    length_unit = 1. if kind=='numerical' else data.get('mass',1.) if kind in ('binary_ks_approx','binary_ks_orbit_approx') else metric.mass
    for key in ("position", "target"):
        camera_data[key] = tuple(length_unit * v for v in camera_data[key])
    camera_data["time"] *= length_unit
    integration_data = asdict(Integration()) | data.get("integration", {})
    geodesics=dict(data.get("geodesics",{}))
    if set(geodesics)-{"integrator","fallback"}:
        raise ValueError("geodesics accepts only integrator and fallback")
    if "integrator" in geodesics: integration_data["integrator"]=geodesics["integrator"]
    if "fallback" in geodesics: integration_data["carter_fallback"]=geodesics["fallback"]
    for key in ("capture_radius", "escape_radius", "max_affine", "max_step", "kerr_source_radius"):
        integration_data[key] *= length_unit
    settings=Integration(**integration_data)
    from .kerr_carter import validate_backend
    validate_backend(metric,settings)
    return metric, Camera(**camera_data), settings


def main():
    parser = argparse.ArgumentParser(description="CPU observatory: Schwarzschild, Kerr, stationary or orbiting experimental binary KS")
    parser.add_argument("--config", type=Path, help="TOML configuration (lengths in M)")
    parser.add_argument("--output", type=Path, default=Path("output"))
    parser.add_argument("--width", type=int, help="override pixel width")
    parser.add_argument("--height", type=int, help="override pixel height")
    parser.add_argument("--background", choices=("starfield", "grid"), help="override background mode")
    parser.add_argument("--method", choices=("ode", "quadrature"), default="ode")
    parser.add_argument("--integrator", choices=("hamiltonian", "carter"), help="override Kerr geodesic backend; default Hamiltonian")
    parser.add_argument("--carter-fallback", choices=("none", "hamiltonian"), help="explicit policy for failed Carter rays")
    parser.add_argument("--samples", type=int, default=1, help="subpixel samples per image axis")
    parser.add_argument("--workers", type=int, default=1,
                        help="CPU processes: 1 serial (default), 0 automatic physical-core selection")
    parser.add_argument("--disk", action=argparse.BooleanOptionalAction, default=None,
                        help="override the optional disk enabled state (--no-disk disables it)")
    args = parser.parse_args()
    try:
        metric, camera, integration = load_configuration(args.config)
        if args.integrator is not None: integration=replace(integration,integrator=args.integrator)
        if args.carter_fallback is not None: integration=replace(integration,carter_fallback=args.carter_fallback)
        from .kerr_carter import validate_backend
        validate_backend(metric,integration)
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
