"""Opt-in hidden disk layers: new trajectories once, independent repaint later."""
import argparse
from collections import Counter
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion

from .background import make_background, linear_to_srgb
from .cli import load_configuration
from .disk import DiskScene, make_disk
from .disk_diagnostics import (load_saved_render, colorize, annotate, crop_bounds,
                               crop_raster, comparison_atlas)
from .display import DiskDisplay, srgb_to_linear
from .emission import PowerLawBlackbody
from .emitter_motion import SchwarzschildCircular
from .geodesic import Integration
from .layer_tracing import CROSSING_DTYPE, LayerScene, layer_task, reference_camera, REGIONS, REASONS
from .metric import Schwarzschild
from .orbital_diagnostics import saved_camera, sampled_camera
from .output import staged_render
from .parallel import scheduled, resolve_workers
from .schwarzschild_orbit import compatible
from .surfaces import ThinAnnulus


def scene_from_metadata(metadata):
    if metadata.get("spacetime",{}).get("type","schwarzschild") != "schwarzschild":
        raise ValueError("Kerr multilayer diagnostics are not supported; use first-hit disk_diagnostics")
    metric = Schwarzschild(metadata["mass"])
    scene = DiskScene(ThinAnnulus(**metadata["geometry"]),
                      SchwarzschildCircular(metric, metadata["geometry"]["normal"], metadata["rotation"]),
                      PowerLawBlackbody(**metadata["emission"]), DiskDisplay(**metadata["display"]))
    return metric, saved_camera(metadata), Integration(**metadata["integration"]), scene


def background_from_metadata(metadata):
    description = metadata["background"]
    grid = description["model"] == "celestial coordinate grid"
    names = ("longitude_degrees", "latitude_degrees", "roll_degrees", "spacing_degrees", "line_width_degrees") if grid else (
        "longitude_degrees", "latitude_degrees", "roll_degrees", "exposure", "resolution")
    return make_background(dict(mode="grid" if grid else "starfield", **{key: description[key] for key in names if key in description}))


def tasks_for_camera(camera, symmetry):
    count = camera.width*camera.height
    if not symmetry:
        return [np.arange(i, min(i+64, count)) for i in range(0, count, 64)]
    full = reference_camera(camera)
    yy, xx = np.indices((camera.height, camera.width))
    ix = 2*(xx+getattr(camera, "origin_x", 0))+1-full.width
    iy = full.height-2*(yy+getattr(camera, "origin_y", 0))-1
    keys = (ix*ix+iy*iy).ravel()
    order = np.argsort(keys, kind="stable")
    borders = np.r_[0, np.flatnonzero(np.diff(keys[order]))+1, count]
    return [order[borders[i]:borders[min(i+4, len(borders)-1)]] for i in range(0, len(borders)-1, 4)]


def collect(metric, camera, settings, scene, max_crossing=4, workers=1, method="auto", progress=None):
    if type(metric) is not Schwarzschild:
        raise ValueError("Kerr multilayer diagnostics are not supported")
    if isinstance(max_crossing, bool) or not isinstance(max_crossing, int) or not 1 <= max_crossing <= np.iinfo(np.uint32).max:
        raise ValueError("max-crossing must be a positive integer")
    workers = resolve_workers(workers)
    if method not in ("auto", "ode", "quadrature"):
        raise ValueError("method must be auto, ode or quadrature")
    eligible = compatible(metric, camera) and np.linalg.norm(camera.position) > scene.surface.outer_radius
    if method == "quadrature" and not eligible:
        raise ValueError("layer quadrature requires a centered static Schwarzschild camera outside the disk; use ode")
    selected = "quadrature" if eligible and method != "ode" else "ode"
    tasks = tasks_for_camera(camera, compatible(metric, camera))
    spec = (metric, camera, settings, scene, max_crossing, selected)
    size = camera.width*camera.height
    rows = [None]*size
    status, reason, backend = np.zeros(size, np.uint8), np.zeros(size, np.uint8), np.zeros(size, np.uint8)
    null, directions, truncated = np.zeros(size), np.full((size, 3), np.nan), np.zeros(size, bool)
    work = np.zeros(3, np.int64)
    statistics = {}
    started = time.perf_counter()
    if workers == 1:
        stream = (layer_task(spec, task) for task in tasks)
    else:
        stream = scheduled(("disk_layers", spec), tasks, workers, statistics)
    from contextlib import closing
    with closing(stream):
        for done, (values, measured) in enumerate(stream, 1):
            work += measured
            for index, records, outcome, why, error, direction, limited, used in values:
                rows[index], status[index], reason[index], null[index], directions[index], truncated[index], backend[index] = records, outcome, why, error, direction, limited, used
            if progress:
                progress(done, len(tasks), time.perf_counter()-started)
    lengths = np.fromiter((len(row) for row in rows), dtype=np.int64, count=size)
    offsets = np.r_[0, np.cumsum(lengths)]
    crossings = np.concatenate(rows) if offsets[-1] else np.empty(0, CROSSING_DTYPE)
    pixels = np.repeat(np.arange(size, dtype=np.int64), lengths)
    arrays = dict(crossings=crossings, pixel_index=pixels, ray_offsets=offsets,
                  ray_status=status.reshape(camera.height, camera.width), ray_reason=reason.reshape(camera.height, camera.width),
                  ray_null_error=null.reshape(camera.height, camera.width), directions=directions.reshape(camera.height, camera.width, 3),
                  crossing_limit_reached=truncated.reshape(camera.height, camera.width), backend=backend.reshape(camera.height, camera.width))
    report = dict(method=selected, requested_method=method, max_crossing=max_crossing, elapsed_seconds=time.perf_counter()-started,
                  workers=workers, execution=statistics, work=dict(geometry_groups_or_integrations=int(work[0]), ode_steps=int(work[1]), rhs_evaluations=int(work[2])),
                  counts={name: int(np.count_nonzero(status == code)) for code, name in ((0, "captured"), (1, "escaped"), (2, "unresolved"))},
                  backend_rays={"quadrature": int(backend.sum()), "ode": int(size-backend.sum())},
                  plane_crossings=int(len(crossings)), valid_emitting_crossings=int(crossings["emission_valid"].sum()),
                  max_recorded_null_error=float(np.max(crossings["null_error"][np.isfinite(crossings["null_error"])])) if np.isfinite(crossings["null_error"]).any() else None,
                  nonfinite_recorded_null_errors=int(np.count_nonzero(~np.isfinite(crossings["null_error"]))),
                  diagnostic_scope="ODE: sampled dense trajectory constraint checks. Quadrature: reconstructed-state null checks, not an ODE path residual.",
                  termination="capture/escape/max_affine as configured. Recording stops at max_crossing; truncated is explicit. ODE cache retains the natural trajectory for diagnostics/reuse.")
    return arrays, report


def layer_grid(data, ordinal):
    """An absent/invalid target crossing is empty; never substitute a later one."""
    shape = data["ray_status"].shape
    size = np.prod(shape)
    records, pixels = data["crossings"], data["pixel_index"]
    chosen = records["plane_ordinal"] == ordinal
    grid = {name: np.full(size, np.nan) for name in ("radius_M", "azimuth_degrees", "g", "bolometric_intensity", "observed_temperature", "phi_radians")}
    valid, concealed = np.zeros(size, bool), np.zeros(size, bool)
    present, geometry_valid = np.zeros(size, bool), np.zeros(size, bool)
    region_codes, emitting_ordinal = np.zeros(size, np.uint8), np.zeros(size, np.uint32)
    intersection_radius = np.full(size, np.nan)
    indices, rec = pixels[chosen], records[chosen]
    selected = rec["emission_valid"]
    present[indices], geometry_valid[indices] = True, rec["geometry_valid"]
    region_codes[indices], emitting_ordinal[indices], intersection_radius[indices] = rec["region"], rec["emitting_ordinal"], rec["radius_M"]
    valid[indices] = selected
    concealed[indices] = selected & (rec["emitting_ordinal"] > 1)
    for name in grid:
        grid[name][indices[selected]] = rec[name][selected]
    return {key: value.reshape(shape) for key, value in grid.items()} | dict(valid=valid.reshape(shape), concealed=concealed.reshape(shape), normally_visible=(valid & ~concealed).reshape(shape),
        plane_present=present.reshape(shape), geometry_valid=geometry_valid.reshape(shape), crossing_region=region_codes.reshape(shape),
        emitting_ordinal=emitting_ordinal.reshape(shape), intersection_radius_M=intersection_radius.reshape(shape))


def first_hits(data):
    size = data["ray_status"].size
    records, pixels = data["crossings"], data["pixel_index"]
    first = np.full(size, len(records), dtype=np.int64)
    valid = records["emission_valid"]
    np.minimum.at(first, pixels[valid], np.flatnonzero(valid))
    return first


def compare_opaque(data, reference, mass=1.):
    records, first = data["crossings"], first_hits(data)
    found = first < len(records)
    plane = np.zeros(first.size, np.uint32)
    plane[found] = records["plane_ordinal"][first[found]]
    plane = plane.reshape(data["ray_status"].shape)
    expected = np.where(reference["status"] == 3, reference["crossing_index"], 0)
    unknown = data["crossing_limit_reached"] & ~found.reshape(plane.shape)
    shared = found.reshape(plane.shape) & (reference["status"] == 3) & ~unknown
    indices = first.reshape(plane.shape)[shared]
    report = dict(first_hit_category_differences=int(np.count_nonzero((plane != expected) & ~unknown)),
                  unknown_first_hit_due_to_recording_limit=int(unknown.sum()),
                  normally_visible_crossing_2_differences=int(np.count_nonzero(layer_grid(data, 2)["normally_visible"] != ((reference["status"] == 3) & (reference["crossing_index"] == 2)))))
    if shared.any():
        denominator = reference["bolometric_intensity"][shared]
        active = denominator > 0
        relative = abs(records["bolometric_intensity"][indices][active]/denominator[active]-1)
        report.update(max_first_hit_position_difference_M=float(np.max(np.linalg.norm(records["event"][indices, 1:]-reference["hit_position"][shared], axis=1)))/mass,
                      max_g_absolute_difference=float(np.max(abs(records["g"][indices]-reference["g"][shared]))),
                      max_relative_bolometric_difference=float(relative.max()) if active.any() else 0.)
    return report


def display_layer(grid, display):
    image = np.zeros(grid["valid"].shape+(3,))
    z = grid["valid"]
    image[z] = display.linear_rgb(grid["bolometric_intensity"][z], grid["observed_temperature"][z])
    return image


def save_rgb(path, linear):
    Image.fromarray(np.uint8(np.rint(255*linear_to_srgb(linear)))).save(path)


def make_views(directory, output=None, target=2, alpha=.35, bounds=None, scale=4, palette=None, features=Path("examples/disk_diagnostics")):
    directory = Path(directory)
    output = directory/"views" if output is None else Path(output)
    metadata = json.loads((directory/"trace_metadata.json").read_text(encoding="utf-8"))
    if not 1 <= target <= metadata["collection"]["max_crossing"]:
        raise ValueError("selected crossing must lie within the recorded max-crossing")
    if not 0 <= alpha <= 1 or not isinstance(scale, int) or scale < 1:
        raise ValueError("alpha must be in [0,1] and scale a positive integer")
    with np.load(directory/"disk_layers.npz", allow_pickle=False) as saved:
        data = dict(saved)
    camera = saved_camera(metadata)
    samples = metadata.get("samples", 1)
    width, height = camera.width, camera.height
    bounds = crop_bounds(bounds, width, height)
    display = DiskDisplay(**(metadata["display"] | ({} if palette is None else dict(palette=palette))))
    output.mkdir(parents=True, exist_ok=True)
    crops = output/"crops"; crops.mkdir(exist_ok=True)
    selected = layer_grid(data, target)
    foreground = layer_grid(data, 1)
    chosen_rgb, first_rgb = display_layer(selected, display), display_layer(foreground, display)
    def image(linear):
        averaged = linear.reshape(height, samples, width, samples, 3).mean(axis=(1, 3))
        return Image.fromarray(np.uint8(np.rint(255*linear_to_srgb(averaged))))
    # Generate all requested layers from the same crossing table.
    available = set(int(v) for v in np.unique(data["crossings"]["plane_ordinal"][data["crossings"]["emission_valid"]]))
    ordinals = sorted(available | set(range(1, min(metadata["collection"]["max_crossing"], 4)+1)) | {target})
    for ordinal in ordinals:
        image(display_layer(layer_grid(data, ordinal), display)).save(output/f"crossing_{ordinal}.png")
    # Reconstruct first-valid-hit opacity from the table as a separate check.
    reconstructed = np.zeros(data["ray_status"].shape+(3,))
    first, records = first_hits(data), data["crossings"]
    found = first < len(records)
    known = found | ~data["crossing_limit_reached"].ravel()
    bg = (data["ray_status"].ravel() == 1) & ~found & known
    sky = background_from_metadata(metadata)
    reconstructed.reshape(-1, 3)[bg] = srgb_to_linear(sky.color(data["directions"].reshape(-1, 3)[bg]))
    hit = records[first[found]]
    reconstructed.reshape(-1, 3)[found] = display.linear_rgb(hit["bolometric_intensity"], hit["observed_temperature"])
    unresolved = ((data["ray_status"].ravel() == 2) & ~found) | ~known
    reconstructed.reshape(-1, 3)[unresolved] = (1, 0, .7)
    opaque_reconstructed = image(reconstructed)
    opaque_reconstructed.save(output/"opaque_reconstructed.png")
    reference_path = directory/"opaque_reference.png"
    if reference_path.exists():
        with Image.open(reference_path) as source:
            opaque = source.convert("RGB")
    else:
        opaque = opaque_reconstructed
    opaque.save(output/"opaque.png")
    weight = alpha*foreground["valid"][..., None]
    composite = image(chosen_rgb*(1-weight)+first_rgb*weight)
    composite.save(output/"composite.png")
    outlined = image(chosen_rgb)
    boundary = foreground["valid"] & ~binary_erosion(foreground["valid"])
    # Display-pixel outline from exact subray membership; no change to data masks.
    outline_pixels = boundary.reshape(height, samples, width, samples).any(axis=(1, 3))
    rgb = np.asarray(outlined).copy(); rgb[outline_pixels] = (245, 245, 245)
    outlined = Image.fromarray(rgb); outlined.save(output/f"crossing_{target}_foreground_outline.png")
    visible = selected["normally_visible"]
    hidden = selected["concealed"]
    categorical = np.full(data["ray_status"].shape+(3,), (24, 29, 37), np.uint8)
    categorical[visible], categorical[hidden] = (0, 180, 230), (240, 90, 160)
    categorical[(data["ray_status"] == 2) & ~selected["valid"]] = (230, 0, 180)
    visibility = Image.fromarray(categorical)
    visibility.save(output/"visibility_pixels.png")
    annotate(visibility, f"Crossing {target}: ordinary opaque visibility", "Cyan=normally visible; pink=concealed by an earlier emitting crossing", "rendered", 0, 1, [],
             footer=["Dark: no valid emitting target crossing; magenta: unresolved/unknown target.",
                     "Prior surfaces are ignored for this diagnostic only; ordinary opacity is unchanged."]).save(output/"visibility.png")
    radius_data = dict(status=np.where(selected["valid"], 3, np.where(data["ray_status"] == 2, 2, 0)), radius_m=selected["radius_M"], radius_valid=selected["valid"],
                       crossing_number=np.full(selected["valid"].shape, target, np.uint32), crossing_valid=selected["valid"],
                       azimuth_degrees=selected["azimuth_degrees"], azimuth_valid=selected["valid"])
    low, high = (metadata["geometry"][key]/metadata["mass"] for key in ("inner_radius", "outer_radius"))
    colors = colorize(radius_data, low, high)
    colors["radius"].save(output/"emission_radius_pixels.png")
    colors["azimuth"].save(output/"emission_azimuth_pixels.png")
    for kind in ("radius", "azimuth"):
        annotate(colors[kind], f"Complete crossing-{target} emitting {kind}", "Includes emission normally hidden by earlier opaque disk intersections", kind, low, high, [],
                 footer=["Black: no valid emitting target crossing, including holes and outer-edge misses.", "Magenta: unresolved target. Geometry and values do not use display brightness."]).save(output/f"emission_{kind}.png")
    Image.fromarray(np.uint8(selected["valid"])*255).save(output/"valid_mask.png")
    region_rgb = np.full(data["ray_status"].shape+(3,), (24, 29, 37), np.uint8)
    for code, rgb_color in ((1, (160, 100, 210)), (2, (0, 180, 220)), (3, (230, 160, 40))):
        region_rgb[selected["crossing_region"] == code] = rgb_color
    region_rgb[selected["plane_present"] & ~selected["geometry_valid"]] = (160, 160, 160)
    annotate(Image.fromarray(region_rgb), f"Crossing {target}: plane intersection region", "Purple=hole; cyan=annulus; orange=beyond outer edge; dark=no recorded target", "rendered", 0, 1, [],
             footer=["Gray: invalid intersection. These are plane-crossing categories, not emitting-hit ordinals."]).save(output/"intersection_region.png")
    np.savez_compressed(output/"target_arrays.npz", **selected, plane_ordinal=target, crop_bounds=np.array(bounds),
                        foreground_crossing_1=foreground["valid"], samples=samples, mass=metadata["mass"])
    x0, y0, x1, y1 = bounds
    np.savez_compressed(crops/"target_arrays.npz", **{name: value[y0*samples:y1*samples, x0*samples:x1*samples] for name, value in selected.items()},
                        plane_ordinal=target, source_crop_bounds=np.array(bounds), samples=samples, mass=metadata["mass"])
    # All presentation crops use one local camera-image box. No retracing here.
    pixel_panels = dict(opaque=opaque, complete_layer=image(chosen_rgb), composite=composite,
                        foreground_outline=outlined, visibility=visibility, emission_radius=colors["radius"], emission_azimuth=colors["azimuth"])
    for name, panel in pixel_panels.items():
        n = 1 if name in ("opaque", "complete_layer", "composite", "foreground_outline") else samples
        crop_raster(panel, bounds, n, scale).save(crops/f"{name}.png")
    # Five requested panels, plus the labelled visibility map, in a 2x3 atlas.
    from PIL import ImageDraw
    from .disk_diagnostics import _font
    panel_size = ((bounds[2]-bounds[0])*2, (bounds[3]-bounds[1])*2)
    atlas = Image.new("RGB", (panel_size[0]*2, (panel_size[1]+26)*3), (246, 246, 248))
    draw = ImageDraw.Draw(atlas)
    names = ("opaque", "complete_layer", "visibility", "composite", "emission_radius", "emission_azimuth")
    titles = ("Original opaque image", f"Complete crossing {target} (prior surfaces ignored)", "Cyan: visible / pink: concealed", f"Crossing 1 alpha={alpha:g} over layer {target}", "Complete layer: emission radius", "Complete layer: emission azimuth")
    for i, (name, title) in enumerate(zip(names, titles)):
        panel = Image.open(crops/f"{name}.png").convert("RGB")
        x, y = (i % 2)*panel_size[0], (i//2)*(panel_size[1]+26)
        draw.text((x+8, y+5), title, fill="black", font=_font(14))
        atlas.paste(panel.resize(panel_size, Image.Resampling.NEAREST), (x, y+26))
    atlas.save(output/"crop_comparison.png")
    report = dict(target_crossing=target, emitting_subrays=int(selected["valid"].sum()), normally_visible_subrays=int(visible.sum()),
                  concealed_subrays=int(hidden.sum()), alpha=alpha, samples=samples, crop_bounds_xyxy_exclusive=list(bounds), enlargement=scale,
                  palette=display.palette, geometry_traced_again=False,
                  composite="Diagnostic alpha blend of separately tone-mapped display-linear RGB; not physical transmission or radiative transfer",
                  ordinal="Plane-crossing ordinal includes hole and outside-edge crossings; emitting ordinal counts valid material intersections only",
                  reference_comparison=metadata.get("opaque_comparison"))
    if reference_path.exists() and opaque.size == opaque_reconstructed.size and palette is None:
        if metadata.get("opaque_reference_display", metadata["display"]) == metadata["display"]:
            difference = abs(np.asarray(opaque).astype(int)-np.asarray(opaque_reconstructed).astype(int))
            report["reconstructed_opaque_image_difference"] = dict(max_8bit=int(difference.max()), differing_pixels=int(np.any(difference, axis=-1).sum()))
    (output/"view_metadata.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    from .layer_analysis import analyze
    analyze(data, selected, metadata, output, target, features)
    return report


def trace_to_directory(metadata, output, workers=1, maximum=4, method="auto", samples=1, reference=None, reference_image=None):
    metric, camera, settings, scene = scene_from_metadata(metadata)
    sampled = sampled_camera(camera, samples)
    last = [0.]
    def progress(done, total, seconds):
        if seconds-last[0] >= 5 or done == total:
            print(f"Layer batches {done}/{total}; elapsed {seconds:.1f}s; estimated remaining {seconds*(total-done)/done:.1f}s", flush=True)
            last[0] = seconds
    def operation(stage):
        arrays, report = collect(metric, sampled, settings, scene, maximum, workers, method, progress)
        np.savez_compressed(stage/"disk_layers.npz", **arrays)
        context = {key: metadata[key] for key in ("mass", "camera", "integration", "geometry", "emission", "display", "rotation", "background", "source_render", "source_window_xyxy", "source_sha256", "opaque_reference_display") if key in metadata}
        context.update(samples=samples, collection=report, crossing_dtype=str(CROSSING_DTYPE),
                       region_codes=REGIONS, reason_codes=REASONS,
                       note="Opt-in hypothetical transparent continuation; normal physical rendering remains first-valid-hit opaque")
        if "counts" in metadata:
            context["source_opaque_counts"] = metadata["counts"]
        if reference is not None and reference["status"].shape == arrays["ray_status"].shape:
            context["opaque_comparison"] = compare_opaque(arrays, reference, metric.mass)
        if reference_image is not None:
            reference_image.save(stage/"opaque_reference.png")
        (stage/"trace_metadata.json").write_text(json.dumps(context, indent=2, allow_nan=False)+"\n", encoding="utf-8")
        return context
    return staged_render(output, operation)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--input-render", type=Path)
    source.add_argument("--config", type=Path)
    source.add_argument("--reuse", type=Path, help="existing multilayer data; no new tracing")
    parser.add_argument("--output", type=Path, default=Path("examples/disk_layer_analysis"))
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--method", choices=("auto", "quadrature", "ode"), default="auto")
    parser.add_argument("--max-crossing", type=int, default=4)
    parser.add_argument("--target", type=int, default=2)
    parser.add_argument("--alpha", type=float, default=.35)
    parser.add_argument("--crop", type=int, nargs=4)
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--width", type=int); parser.add_argument("--height", type=int)
    parser.add_argument("--samples", type=int, help="samples per output pixel axis (default: match input render, otherwise 1)")
    parser.add_argument("--window", type=int, nargs=4, help="native source-camera bounds, preserving pixel-to-ray geometry")
    parser.add_argument("--density", type=int, default=1, help="new rays per original pixel width, only with --window")
    parser.add_argument("--palette", choices=("thermal", "blackbody", "gray"), help="display-only override")
    parser.add_argument("--features", type=Path, default=Path("examples/disk_diagnostics"), help="earlier masks for reference-case continuity analysis")
    args = parser.parse_args()
    try:
        if args.reuse:
            report = make_views(args.reuse, args.output/"views", args.target, args.alpha, args.crop, args.scale, args.palette, args.features)
        else:
            reference = original = None
            source_samples = 1
            if args.config:
                import tomllib
                metric, camera, settings = load_configuration(args.config)
                if type(metric) is not Schwarzschild:
                    raise ValueError("Kerr multilayer diagnostics are not supported")
                config = tomllib.loads(args.config.read_text(encoding="utf-8"))
                scene = make_disk(metric, config.get("disk"))
                if scene is None:
                    raise ValueError("a luminous disk must be enabled for layer diagnostics")
                metadata = dict(mass=metric.mass, camera=asdict(camera), integration=asdict(settings),
                                geometry=asdict(scene.surface), emission=asdict(scene.emission), display=asdict(scene.display),
                                rotation=scene.motion.rotation, background=make_background(config.get("background")).description)
            else:
                source_path = (args.input_render or Path("output/disk-edge-on-blackbody-hires")).resolve()
                destination = args.output.resolve()
                if destination == source_path or source_path in destination.parents or destination in source_path.parents:
                    raise ValueError("layer output must be separate from the original render")
                metadata, reference, original, _, source_samples = load_saved_render(source_path)
                metadata["source_render"] = str(source_path)
                metadata["opaque_reference_display"] = dict(metadata["display"])
                metadata["source_sha256"] = {name: hashlib.sha256((source_path/name).read_bytes()).hexdigest()
                                              for name in ("diagnostics.json", "disk_rays.npz", "shadow.png")}
            camera = saved_camera(metadata)
            if args.width is not None or args.height is not None:
                camera = replace(camera, width=camera.width if args.width is None else args.width,
                                 height=camera.height if args.height is None else args.height)
                reference = original = None
            if args.window:
                from .diagnostic_zoom import window_camera
                camera = window_camera(camera, tuple(args.window), args.density)
                metadata["source_window_xyxy"] = args.window
                reference = None
                original = original.crop(tuple(args.window)) if original is not None and args.density == 1 else None
            elif args.density != 1:
                raise ValueError("--density requires --window")
            samples = source_samples if args.samples is None else args.samples
            if samples < 1:
                raise ValueError("samples must be positive")
            if not 1 <= args.target <= args.max_crossing:
                raise ValueError("target must be between 1 and max-crossing")
            if not 0 <= args.alpha <= 1 or args.scale < 1:
                raise ValueError("alpha must be in [0,1] and scale positive")
            crop_bounds(args.crop, camera.width, camera.height)
            metadata["camera"] = asdict(camera)
            if args.palette:
                metadata["display"] = metadata["display"] | dict(palette=args.palette)
            trace_to_directory(metadata, args.output, args.workers, args.max_crossing, args.method, samples, reference, original)
            report = make_views(args.output, target=args.target, alpha=args.alpha, bounds=args.crop, scale=args.scale, features=args.features)
        print(json.dumps(report, indent=2))
    except KeyboardInterrupt:
        parser.exit(130, "Layer tracing interrupted; workers stopped.\n")
    except (ValueError, OSError, KeyError, RuntimeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
