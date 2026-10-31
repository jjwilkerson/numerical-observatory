"""Saved-hit accumulated-angle maps and explicitly transparent sky geometry."""
import argparse
from collections import Counter
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw

from .camera import Camera
from .disk_diagnostics import (load_saved_render, extract, colorize, radius_color,
                               crop_bounds, crop_raster, comparison_atlas, annotate, _font)
from .metric import Schwarzschild
from .orbital_angle import OrbitalAngle, orbital_frame, alignment_rings, AmbiguousAngle
from .schwarzschild_orbit import compatible


def saved_camera(metadata):
    values = metadata["camera"]
    if "reference_width" in values:
        from .diagnostic_zoom import WindowCamera
        return WindowCamera(**values)
    return Camera(**values)


def sampled_camera(camera, samples):
    extra = {}
    if hasattr(camera, "reference_width"):
        extra = {key: getattr(camera, key)*samples for key in ("reference_width", "reference_height", "origin_x", "origin_y")}
    return replace(camera, width=camera.width*samples, height=camera.height*samples, **extra)


def compute_angles(metadata, raw, samples=1, eps=1e-11, progress=None):
    if metadata.get("spacetime",{}).get("type","schwarzschild") != "schwarzschild":
        raise ValueError("Schwarzschild planar orbital angles and alignment rings are not valid for Kerr")
    camera = sampled_camera(saved_camera(metadata), samples)
    metric = Schwarzschild(metadata["mass"])
    if not compatible(metric, camera):
        raise ValueError("saved-data angle reconstruction requires a centered coordinate-static Schwarzschild camera; no automatic ODE fallback is launched")
    disk, _ = extract(raw, metadata)
    status = disk["status"]
    if status.shape != (camera.height, camera.width):
        raise ValueError("saved ray-grid/camera dimensions disagree")
    h, w = status.shape
    yy, xx = np.indices(status.shape)
    reference_w = getattr(camera, "reference_width", w)
    reference_h = getattr(camera, "reference_height", h)
    ix = 2*(xx+getattr(camera, "origin_x", 0))+1-reference_w
    iy = reference_h-2*(yy+getattr(camera, "origin_y", 0))-1
    keys, reverse = np.unique(ix*ix+iy*iy, return_inverse=True)
    reverse = reverse.reshape(status.shape)
    f, right, up = camera.axes()
    initial_radial = np.asarray(camera.position)/np.linalg.norm(camera.position)
    sky = np.full(len(keys), np.nan)
    outcomes = np.full(len(keys), 2, dtype=np.uint8)
    phi_emit, residual = np.full(status.shape, np.nan), np.full(status.shape, np.nan)
    branch_codes = np.zeros(status.shape, dtype=np.uint8)  # 1=inward, 2=outward
    reasons = Counter()
    position = raw.get("hit_position")
    positions = None if position is None else np.asarray(position).reshape(-1, 3)
    groups = {}
    for index in np.flatnonzero(status.ravel() == 3):
        groups.setdefault(int(reverse.ravel()[index]), []).append(int(index))
    started = time.perf_counter()
    for i, key in enumerate(keys):
        offset = np.sqrt(key)*np.tan(np.deg2rad(camera.fov_degrees)/2)/reference_w
        momentum = camera.momentum(metric, f+offset*right)
        radius, impact, speed, radial, tangent = orbital_frame(metric, camera.event, momentum)
        trajectory = OrbitalAngle(metric.mass, radius, impact, speed < 0, eps)
        outcomes[i] = {"captured": 0, "escaped": 1, "unresolved": 2}[trajectory.outcome]
        if trajectory.background is not None:
            sky[i] = trajectory.background
        for index in groups.get(i, ()):
            if positions is None or not np.isfinite(positions[index]).all():
                reasons["missing_hit_position"] += 1
                continue
            row, col = divmod(index, w)
            if key:
                transverse = (ix[row, col]*right+iy[row, col]*up)/np.sqrt(key)
                rotation = (np.outer(initial_radial, initial_radial)+np.outer(transverse, right)
                            +np.outer(np.cross(initial_radial, transverse), np.cross(initial_radial, right)))
                actual_tangent = None if tangent is None else rotation @ tangent
            else:
                actual_tangent = None
            try:
                angle, branch, error = trajectory.saved_hit(positions[index], radial, actual_tangent)
            except AmbiguousAngle:
                reasons["ambiguous_radial_leg"] += 1
                continue
            except ValueError:
                reasons["inconsistent_saved_hit"] += 1
                continue
            phi_emit[row, col], residual[row, col] = angle, error
            branch_codes[row, col] = 1 if branch == "inward" else 2
        if progress and (i % 2000 == 0 or i == len(keys)-1):
            progress(i+1, len(keys), time.perf_counter()-started)
    bg_status = outcomes[reverse]
    phi_background = sky[reverse]
    mismatch = ((status == 0) & (bg_status == 1)) | ((status == 1) & (bg_status == 0))
    # A physically captured saved ray never receives an escaping sky angle.
    phi_background[(status == 0) | (status == 2) | ~np.isin(status, [0, 1, 2, 3]) | mismatch] = np.nan
    bg_status[status == 0] = 0
    bg_status[status == 2] = 2
    bg_status[~np.isin(status, [0, 1, 2, 3]) | ((status == 1) & mismatch)] = 99
    arrays = dict(source_status=status, background_status=bg_status,
                  phi_emit_radians=phi_emit, phi_emit_over_pi=phi_emit/np.pi,
                  phi_background_radians=phi_background, phi_background_over_pi=phi_background/np.pi,
                  phi_emit_valid=np.isfinite(phi_emit), phi_background_valid=np.isfinite(phi_background),
                  hypothetical_transparent_disk=(status == 3) & np.isfinite(phi_background),
                  physically_visible_background=(status == 1) & np.isfinite(phi_background),
                  emission_radial_leg=branch_codes, saved_hit_angular_residual_radians=residual,
                  image_radius_pixels=np.sqrt(ix*ix+iy*iy)/(2*samples))
    details = dict(distinct_conserved_orbits=len(keys), elapsed_seconds=time.perf_counter()-started,
                   invalid_emission_reasons=dict(reasons), classification_mismatch_count=int(mismatch.sum()),
                   max_saved_hit_geometry_residual_radians=float(np.nanmax(residual)) if np.isfinite(residual).any() else None,
                   emission_branch_counts={name: int(np.count_nonzero(branch_codes == code)) for code, name in ((1, "inward"), (2, "outward"))},
                   quadrature_tolerance=eps, saved_hit_geometry_check_tolerance_radians=5e-6,
                   reconstruction="camera tetrad -> canonical initial p_t and |x cross p|; exact rotational reuse; partial radial quadratures checked against saved orbital-plane position, never disk crossing number or disk azimuth")
    return arrays, details


RING_STYLES = [("shadow", (255, 255, 255), "dash"), (1, (255, 160, 0), "dash"),
               (2, (255, 40, 220), "solid"), (3, (60, 255, 130), "dot")]


def ring_overlay(image, rings):
    """Fractional native-pixel radii; no displacement of unresolved rings."""
    scale = 8
    layer = Image.new("RGBA", (image.width*scale, image.height*scale))
    draw = ImageDraw.Draw(layer)
    cx, cy = rings["center_xy"]
    center = ((cx+.5)*scale-.5, (cy+.5)*scale-.5)
    for order, color, style in RING_STYLES:
        radius = rings["shadow"]["radius_pixels"] if order == "shadow" else next(r["radius_pixels"] for r in rings["rings"] if r["phase_pi"] == order)
        r = radius*scale
        box = (center[0]-r, center[1]-r, center[0]+r, center[1]+r)
        if style == "solid":
            draw.ellipse(box, outline=color+(220,), width=6)
        else:
            step, visible = (14, 8) if style == "dash" else (8, 2)
            for angle in range(0, 360, step):
                draw.arc(box, angle, min(360, angle+visible), fill=color+(230,), width=6)
    layer = layer.resize(image.size, Image.Resampling.LANCZOS)
    return Image.alpha_composite(image.convert("RGBA"), layer).convert("RGB")


def numbers(values):
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    return dict(count=int(values.size), minimum=float(values.min()), maximum=float(values.max()), median=float(np.median(values))) if values.size else dict(count=0, minimum=None, maximum=None, median=None)


def feature_statistics(arrays, disk, features, rings):
    result = {}
    second = next(r["radius_pixels"] for r in rings["rings"] if r["phase_pi"] == 2)
    selections = (("upper_halo", 2), ("left_branch", 2), ("right_branch", 2), ("lower_halo", 3))
    for name, crossing in selections:
        mask = features[name] & disk["crossing_valid"] & (disk["crossing_number"] == crossing)
        if mask.shape != arrays["source_status"].shape:
            raise ValueError("feature mask does not match the angle grid")
        selected = mask & arrays["phi_emit_valid"]
        radius = arrays["image_radius_pixels"][selected]
        phi = arrays["phi_emit_over_pi"]
        bg = arrays["phi_background_over_pi"]
        record = dict(recorded_crossing=crossing, selected_samples=int(mask.sum()),
                      phi_emit_over_pi=numbers(phi[selected]), phi_background_over_pi=numbers(bg[selected]),
                      signed_distance_from_shadow_pixels=numbers(radius-rings["shadow"]["radius_pixels"]),
                      signed_distance_from_second_ring_pixels=numbers(radius-second),
                      absolute_distance_from_second_ring_pixels=numbers(abs(radius-second)), representatives=[])
        locations = np.flatnonzero(selected)
        if locations.size:
            values = phi.ravel()[locations]
            picks = [np.argmin(values), np.argmax(values), np.argmin(abs(values-np.median(values)))]
            for index in dict.fromkeys(int(locations[pick]) for pick in picks):
                y, x = np.unravel_index(index, selected.shape)
                record["representatives"].append(dict(x=int(x), y=int(y), phi_emit_over_pi=float(phi[y, x]),
                    phi_background_over_pi=float(bg[y, x]) if np.isfinite(bg[y, x]) else None,
                    distance_shadow_pixels=float(arrays["image_radius_pixels"][y, x]-rings["shadow"]["radius_pixels"]),
                    distance_second_ring_pixels=float(arrays["image_radius_pixels"][y, x]-second)))
        if "branch" in name:
            rows = []
            for y in np.unique(np.where(selected)[0]):
                xs = np.flatnonzero(selected[y])
                rows.append(dict(y=int(y), x=xs.tolist(), phi_emit_over_pi=phi[y, xs].tolist(),
                                 phi_background_over_pi=bg[y, xs].tolist(), radius_pixels=arrays["image_radius_pixels"][y, xs].tolist()))
            record["rows"] = rows
        result[name] = record
    return result


def angle_map(values, valid, status, maximum):
    placeholder = dict(status=status, radius_valid=valid, radius_m=values,
                       crossing_valid=np.zeros(status.shape, bool), crossing_number=np.zeros(status.shape, np.uint32),
                       azimuth_valid=np.zeros(status.shape, bool), azimuth_degrees=np.full(status.shape, np.nan))
    return colorize(placeholder, 0, maximum)["radius"]


def generate(source, output, features=None, bounds=(179, 134, 461, 346), scale=4, validate_ode=False):
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("orbital output must be separate from the source render")
    if features is not None:
        features = Path(features).resolve()
        if output == features or features in output.parents or output in features.parents:
            raise ValueError("orbital output must be separate from previous diagnostic products")
    metadata, raw, original, ray_path, samples = load_saved_render(source)
    if metadata.get("spacetime",{}).get("type","schwarzschild") != "schwarzschild":
        raise ValueError("Schwarzschild planar orbital angles and alignment rings are not valid for Kerr")
    bounds = crop_bounds(bounds, *original.size)
    if not isinstance(scale, int) or scale < 1:
        raise ValueError("scale must be a positive integer")
    disk, disk_details = extract(raw, metadata)
    camera = saved_camera(metadata)
    rings = alignment_rings(Schwarzschild(metadata["mass"]), camera)
    arrays, calculation = compute_angles(metadata, raw, samples, progress=lambda n, total, seconds:
                                          print(f"Orbital quadratures {n}/{total}; elapsed {seconds:.1f}s", flush=True))
    output.mkdir(parents=True, exist_ok=True)
    crops = output/"crops"; crops.mkdir(exist_ok=True)
    np.savez_compressed(output/"orbital_angles.npz", **arrays, samples=samples, crop_bounds=np.array(bounds))
    x0, y0, x1, y1 = bounds
    np.savez_compressed(crops/"orbital_angles.npz", **{key: value[y0*samples:y1*samples, x0*samples:x1*samples] for key, value in arrays.items()},
                        samples=samples, source_crop_bounds=np.array(bounds))
    maxima = [np.nanmax(v) for v in (arrays["phi_emit_over_pi"], arrays["phi_background_over_pi"]) if np.isfinite(v).any()]
    maximum = max(1, int(np.ceil(max(maxima)))) if maxima else 1
    maps = dict(phi_emit=angle_map(arrays["phi_emit_over_pi"], arrays["phi_emit_valid"], arrays["source_status"], maximum),
                phi_background=angle_map(arrays["phi_background_over_pi"], arrays["phi_background_valid"], arrays["background_status"], maximum))
    crossing = colorize(disk, 6, 20)["crossing"]
    # Overlay native output pixels before mapping them to a supersampled grid.
    crossing_native = crossing.resize(original.size, Image.Resampling.NEAREST)
    overlays = dict(visible_alignment_rings=ring_overlay(original, rings), crossing_alignment_rings=ring_overlay(crossing_native, rings))
    titles = dict(phi_emit="Accumulated orbital angle to opaque emission: Phi_emit / pi",
                  phi_background="Background orbital angle / pi: TRANSPARENT-DISK geometry",
                  visible_alignment_rings="Disk image with celestial alignment rings (background to infinity)",
                  crossing_alignment_rings="Disk-plane crossing map with celestial alignment rings")
    footer = ["Black: no escaping continuation; magenta: unresolved; gray: unavailable angle.",
              "Disk-hit sky angles are HYPOTHETICAL transparent continuations, not disk light."]
    legend = ["White dashed: critical shadow; orange dashed: Phi_bg=pi.",
              "Magenta: Phi_bg=2pi; green dotted: Phi_bg=3pi. Higher rings not spread apart."]
    panels = []
    for name in ("visible_alignment_rings", "crossing_alignment_rings", "phi_emit", "phi_background"):
        image = maps[name] if name in maps else overlays[name].resize((original.width*samples, original.height*samples), Image.Resampling.NEAREST)
        image.save(output/f"{name}_pixels.png")
        is_angle = name in maps
        annotate(image, titles[name], "Saved camera rays; angle=0 at observer; nonnegative and unwrapped", "radius" if is_angle else "rendered",
                 0, maximum, [], unit="", footer=(footer if name == "phi_background" else None) if is_angle else legend).save(output/f"{name}.png")
        cropped = crop_raster(image, bounds, samples, scale)
        cropped.save(crops/f"{name}_pixels.png")
        annotate(cropped, titles[name], f"Source bounds {list(bounds)} exclusive; {scale}x nearest-neighbor magnification, no new rays",
                 "radius" if is_angle else "rendered", 0, maximum, [], unit="",
                 footer=(footer if name == "phi_background" else None) if is_angle else legend).save(crops/f"{name}.png")
        panels.append(cropped)
    comparison_atlas(panels, ("Visible image + calculated rings", "Crossing categories + calculated rings", "Phi_emit / pi", "Phi_background / pi (transparent disk)"),
                     ((x1-x0)*samples*2, (y1-y0)*samples*2)).save(output/"crop_comparison.png")
    feature_result = None
    if features is not None and (features/"feature_masks.npz").exists():
        with np.load(features/"diagnostic_arrays.npz") as previous:
            for key in ("status", "radius_m", "crossing_number", "azimuth_degrees"):
                try:
                    np.testing.assert_array_equal(previous[key], disk[key])
                except AssertionError as exc:
                    raise ValueError("previous feature masks belong to different ray data; supply matching --features or --no-features") from exc
        with np.load(features/"feature_masks.npz") as masks:
            feature_result = feature_statistics(arrays, disk, masks, rings)
        (output/"feature_statistics.json").write_text(json.dumps(feature_result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    (output/"alignment_rings.json").write_text(json.dumps(rings, indent=2)+"\n", encoding="utf-8")
    validation = None
    if validate_ode:
        from .orbital_validation import validate_render
        validation = validate_render(metadata, raw, arrays)
        (output/"ode_validation.json").write_text(json.dumps(validation, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    record = dict(source_render=str(source), source_rays=str(ray_path), camera=metadata["camera"], mass=metadata["mass"],
                  source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (source/"diagnostics.json", source/"shadow.png", ray_path)},
                  source_image_size=list(original.size), samples=samples, crop_bounds_xyxy_exclusive=list(bounds), enlargement=scale,
                  angle_convention="nonnegative unwrapped Schwarzschild orbital angle; Phi=0 at camera; dPhi/dlambda=|L|/r^2 along past-directed ray",
                  background_convention="infinity, including analytic tail past finite escape sphere; disk-hit values are hypothetical transparent-disk continuations",
                  array_units="radians and radians/pi are both saved; signed image distances use native output pixels",
                  emission_leg_codes={"0": "unavailable", "1": "inward", "2": "outward"},
                  calculation=calculation, disk_extraction=disk_details,
                  ode_image_retracing=False, optional_ode_audit_cases=0 if validation is None else len(validation["cases"]),
                  optional_ode_integrations=0 if validation is None else validation["ode_integrations"],
                  features_source=None if features is None else str(features), map_scale_pi=[0, maximum],
                  overlay_styles=[dict(target=str(order), rgb=list(color), style=style) for order, color, style in RING_STYLES],
                  overlay_rasterization="8x antialiased geometry, native center/radii, no artificial ring separation",
                  source_status_codes={"0": "captured", "1": "escaped", "2": "unresolved", "3": "opaque disk emission"})
    record["background_status_codes"] = {"0": "no escaping continuation", "1": "escaping geometry", "2": "unresolved", "99": "invalid reconstruction/source status"}
    (output/"metadata.json").write_text(json.dumps(record, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("output/disk-edge-on-blackbody-hires"))
    parser.add_argument("--output", type=Path, default=Path("examples/orbital_diagnostics"))
    parser.add_argument("--features", type=Path, default=Path("examples/disk_diagnostics"))
    parser.add_argument("--no-features", action="store_true", help="generate maps without prior feature selections")
    parser.add_argument("--crop", type=int, nargs=4, default=(179, 134, 461, 346))
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--validate-ode", action="store_true", help="explicit small independent ODE audit; never rerenders the image")
    args = parser.parse_args()
    try:
        generate(args.input, args.output, None if args.no_features else args.features, args.crop, args.scale, args.validate_ode)
    except (ValueError, OSError, KeyError) as exc:
        parser.error(str(exc))
    print(f"Saved accumulated-angle diagnostics in {args.output}")


if __name__ == "__main__":
    main()
