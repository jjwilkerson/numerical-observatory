"""Postprocess first opaque disk hits; never import or invoke a ray tracer."""
import argparse
import colorsys
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


# Nine-anchor approximation to the perceptually ordered viridis ramp.
RADIUS_COLORS = np.array([[68, 1, 84], [71, 44, 122], [59, 81, 139],
                         [44, 113, 142], [33, 144, 141], [39, 173, 129],
                         [92, 200, 99], [170, 220, 50], [253, 231, 37]])
CATEGORIES = [(230, 159, 0), (0, 158, 210), (204, 121, 167), (0, 180, 90),
              (240, 228, 66), (0, 114, 178), (213, 94, 0), (86, 180, 233)]
BACKGROUND = (24, 29, 37)
MISSING = (160, 160, 160)


def disk_basis(normal):
    """Zero is projected global +X (fallback +Y); positive follows n cross e0."""
    n = np.asarray(normal, dtype=float)
    if n.shape != (3,) or not np.isfinite(n).all() or np.linalg.norm(n) == 0:
        raise ValueError("disk normal must be a finite nonzero three-vector")
    n = n/np.linalg.norm(n)
    reference = np.array([1., 0., 0.])
    e0 = reference-n*np.dot(reference, n)
    if np.linalg.norm(e0) < 1e-10:
        reference = np.array([0., 1., 0.])
        e0 = reference-n*np.dot(reference, n)
    e0 /= np.linalg.norm(e0)
    return n, e0, np.cross(n, e0)


def extract(data, metadata):
    """Preserve the saved ray grid. No averaging of categories or angles."""
    if "status" not in data:
        raise ValueError("saved status array is required to identify emitted rays")
    status = np.asarray(data["status"])
    if status.ndim != 2:
        raise ValueError("status must be a two-dimensional ray grid")
    shape = status.shape
    mass = metadata.get("mass")
    if mass is None or not np.isfinite(mass) or mass <= 0:
        raise ValueError("diagnostics.json must contain a positive mass to express radii in M")
    def field(name, tail=()):
        if name not in data:
            return None
        value = np.asarray(data[name])
        if value.shape != shape+tail:
            raise ValueError(f"{name} shape does not match the saved ray grid")
        return value
    emitted = status == 3
    radial = lambda positions: np.linalg.norm(positions, axis=-1)
    if metadata.get("spacetime", {}).get("type") == "kerr":
        from .kerr import Kerr
        metric = Kerr(mass, metadata["spacetime"]["spin"])
        def radial(positions):
            flat = np.asarray(positions).reshape(-1,3)
            values = np.full(len(flat), np.nan)
            for i in np.flatnonzero(np.isfinite(flat).all(axis=1)):
                try:
                    values[i] = metric.radius(np.r_[0.,flat[i]])
                except ValueError:
                    pass
            return values.reshape(np.asarray(positions).shape[:-1])
    position = field("hit_position", (3,))
    radius = field("hit_radius")
    warnings = []
    radius_source = "hit_radius"
    if radius is None and position is not None:
        radius = radial(position)
        radius_source = ("Kerr r(hit_position)" if metadata.get("spacetime",{}).get("type")=="kerr"
                         else "norm(hit_position)")
    if radius is None:
        radius = np.full(shape, np.nan)
        warnings.append("Emission radius unavailable: neither hit_radius nor hit_position was saved.")
    radius_valid = emitted & np.isfinite(radius) & (radius > 0)
    radius_m = np.where(radius_valid, radius/mass, np.nan)
    crossing = field("crossing_index")
    if crossing is None:
        crossing = np.full(shape, np.nan)
        warnings.append("Crossing number unavailable: crossing_index was not saved; no category is inferred.")
    crossing_valid = (emitted & np.isfinite(crossing) & (crossing >= 1)
                      & (crossing == np.floor(crossing)) & (crossing <= np.iinfo(np.uint32).max))
    crossing_number = np.where(crossing_valid, crossing, 0).astype(np.uint32)
    azimuth = np.full(shape, np.nan)
    azimuth_valid = np.zeros(shape, dtype=bool)
    normal = metadata.get("geometry", {}).get("normal")
    basis = None
    if position is None or normal is None:
        warnings.append("Emission azimuth unavailable: hit_position and geometry.normal are both required.")
    else:
        n, e0, e90 = disk_basis(normal)
        basis = dict(normal=n.tolist(), zero_direction=e0.tolist(), positive_90_direction=e90.tolist())
        norm = np.linalg.norm(position, axis=-1)
        in_plane = np.abs(position @ n) <= 1e-6*np.maximum(norm, mass)
        azimuth_valid = emitted & np.isfinite(position).all(axis=-1) & (norm > 0) & in_plane
        azimuth[azimuth_valid] = np.degrees(np.arctan2(position[azimuth_valid] @ e90,
                                                      position[azimuth_valid] @ e0)) % 360
        if np.any(emitted & np.isfinite(position).all(axis=-1) & ~in_plane):
            warnings.append("Some saved emitting positions are off the disk plane; their azimuths are invalid.")
    if position is not None and radius_valid.any():
        common = radius_valid & np.isfinite(position).all(axis=-1)
        discrepancy = float(np.max(np.abs(radial(position[common])-radius[common]))) if common.any() else None
    else:
        discrepancy = None
    arrays = dict(status=status.copy(), emitted=emitted, radius_m=radius_m,
                  radius_valid=radius_valid, crossing_number=crossing_number,
                  crossing_valid=crossing_valid, azimuth_degrees=azimuth, azimuth_valid=azimuth_valid)
    details = dict(radius_source=radius_source, basis=basis, warnings=warnings,
                   max_saved_radius_position_disagreement=discrepancy,
                   valid_counts={name: int(arrays[name+"_valid"].sum()) for name in ("radius", "crossing", "azimuth")},
                   missing_emitted_counts={name: int((emitted & ~arrays[name+"_valid"]).sum()) for name in ("radius", "crossing", "azimuth")})
    return arrays, details


def radius_color(values, low, high):
    t = np.clip((np.asarray(values)-low)/(high-low), 0, 1)
    return np.stack([np.interp(t, np.linspace(0, 1, len(RADIUS_COLORS)), RADIUS_COLORS[:, i])
                     for i in range(3)], axis=-1).round().astype(np.uint8)


def azimuth_color(degrees):
    # Hue is periodic; values at exactly 0 and 360 are bit-identical.
    hue = (np.asarray(degrees) % 360)/60
    sector = np.floor(hue).astype(int) % 6
    x = 1-np.abs(hue % 2-1)
    zero, one = np.zeros_like(x), np.ones_like(x)
    choices = np.array([[one, x, zero], [x, one, zero], [zero, one, x],
                        [zero, x, one], [x, zero, one], [one, zero, x]])
    rgb = np.moveaxis(choices, 1, -1)
    return np.rint(255*(.12+.83*rgb[sector, ...] if hue.ndim == 0 else
                         .12+.83*np.take_along_axis(rgb, sector[None, ..., None], axis=0)[0])).astype(np.uint8)


def category_color(number):
    if number <= len(CATEGORIES):
        return CATEGORIES[number-1]
    return tuple(round(255*c) for c in colorsys.hsv_to_rgb((number*.61803398875) % 1, .7, .95))


def colorize(arrays, low, high):
    status = arrays["status"]
    yy, xx = np.indices(status.shape)
    hatch = ((xx+yy)//3) % 2 == 0
    base = np.full(status.shape+(3,), MISSING, dtype=np.uint8)
    base[status == 0] = (0, 0, 0)
    base[status == 1] = BACKGROUND
    base[status == 2] = (230, 0, 180)
    base[(status == 2) & hatch] = (80, 0, 60)
    base[(status == 3) & hatch] = (100, 100, 100)
    maps = {}
    for name in ("radius", "crossing", "azimuth"):
        rgb = base.copy()
        valid = arrays[name+"_valid"]
        if name == "radius":
            rgb[valid] = radius_color(arrays["radius_m"][valid], low, high)
        elif name == "azimuth":
            rgb[valid] = azimuth_color(arrays["azimuth_degrees"][valid])
        else:
            for number in np.unique(arrays["crossing_number"][valid]):
                rgb[valid & (arrays["crossing_number"] == number)] = category_color(int(number))
        maps[name] = Image.fromarray(rgb)
    return maps


def crop_bounds(bounds, width, height):
    if bounds is None:
        bounds = (int(.28*width), int(.28*height), int(np.ceil(.72*width)), int(np.ceil(.72*height)))
    if len(bounds) != 4 or any(int(v) != v for v in bounds):
        raise ValueError("crop requires four integer bounds: x0 y0 x1 y1")
    x0, y0, x1, y1 = map(int, bounds)
    if not 0 <= x0 < x1 <= width or not 0 <= y0 < y1 <= height:
        raise ValueError("crop must lie within the source image; upper bounds are exclusive")
    return x0, y0, x1, y1


def _font(size):
    # Pillow >=10 supports a portable scalable default without external fonts.
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow 10.0/10.1 compatibility
        return ImageFont.load_default()


def crop_raster(image, bounds, samples=1, scale=4):
    """Shared half-open source-pixel crop, retaining the exact subray grid."""
    x0, y0, x1, y1 = bounds
    box = tuple(value*samples for value in bounds)
    return image.crop(box).resize(((x1-x0)*samples*scale, (y1-y0)*samples*scale), Image.Resampling.NEAREST)


def comparison_atlas(panels, titles, panel_size):
    atlas = Image.new("RGB", (panel_size[0]*2, (panel_size[1]+26)*2), (246, 246, 248))
    draw = ImageDraw.Draw(atlas)
    for i, (panel, title) in enumerate(zip(panels, titles)):
        px, py = (i % 2)*panel_size[0], (i//2)*(panel_size[1]+26)
        draw.text((px+8, py+5), title, fill="black", font=_font(14))
        atlas.paste(panel.resize(panel_size, Image.Resampling.NEAREST), (px, py+26))
    return atlas


def annotate(image, title, subtitle, kind, low, high, categories, unit="M", footer=None):
    width, height = image.size
    canvas = Image.new("RGB", (max(width+200, 720), max(height+140, 320)), (246, 246, 248))
    canvas.paste(image, (20, 55))
    draw = ImageDraw.Draw(canvas)
    draw.text((20, 8), title, fill="black", font=_font(18))
    draw.text((20, 34), subtitle, fill=(45, 45, 50), font=_font(12))
    x = width+45
    if kind in ("radius", "azimuth"):
        bar_height = max(100, min(height, 360))
        values = np.linspace(high, low, bar_height) if kind == "radius" else np.linspace(360, 0, bar_height)
        colors = radius_color(values, low, high) if kind == "radius" else azimuth_color(values)
        bar = Image.fromarray(np.repeat(colors[:, None], 24, axis=1))
        canvas.paste(bar, (x, 55))
        ticks = np.linspace(low, high, 5) if kind == "radius" else np.arange(0, 361, 90)
        for value in ticks:
            fraction = (value-low)/(high-low) if kind == "radius" else value/360
            y = 55+round((1-fraction)*(bar_height-1))
            draw.line((x+24, y, x+29, y), fill="black")
            suffix = (f" {unit}" if unit else "") if kind == "radius" else " deg"
            draw.text((x+33, y-5), f"{value:g}"+suffix, fill="black", font=_font(12))
    elif kind == "crossing":
        for i, number in enumerate(categories):
            y = 60+i*24
            if y+22 >= canvas.height:
                expanded = Image.new("RGB", (canvas.width, y+60), (246, 246, 248))
                expanded.paste(canvas, (0, 0)); canvas = expanded; draw = ImageDraw.Draw(canvas)
            draw.rectangle((x, y, x+18, y+15), fill=category_color(number))
            draw.text((x+26, y), f"Crossing {number}", fill="black", font=_font(12))
    y = height+70
    if footer is not None:
        for i, line in enumerate(footer):
            draw.text((20, y+19*i), line, fill="black", font=_font(12))
    elif kind == "rendered":
        draw.text((20, y), "Original saved display pixels; no scientific values inferred from their colors.", fill="black", font=_font(12))
    else:
        draw.text((20, y), "Black: captured   Dark blue-gray: escaped/background", fill="black", font=_font(12))
        draw.text((20, y+19), "Magenta hatch: unresolved   Gray hatch: missing/invalid emitted value", fill="black", font=_font(12))
        draw.text((20, y+38), "Other status codes: gray. Raster origin is top-left; x right, y down.", fill="black", font=_font(12))
    return canvas


def load_saved_render(source, rays=None):
    """Shared read-only loader for disk-hit and orbital-angle postprocessors."""
    source = Path(source).resolve()
    metadata_path, ray_path, image_path = source/"diagnostics.json", source/"disk_rays.npz", source/"shadow.png"
    if rays is not None:
        ray_path = Path(rays).resolve()
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    # Grid/background companions can reference saved data in another directory.
    if rays is None and not ray_path.exists() and metadata.get("ray_data_source"):
        target = Path(metadata["ray_data_source"])
        if not target.is_absolute():
            target = source/target
        ray_path = target if target.suffix == ".npz" else target/"disk_rays.npz"
    with np.load(ray_path, allow_pickle=False) as data:
        raw = {key: data[key] for key in data.files}
        samples = int(data["samples"]) if "samples" in data else int(metadata.get("samples_per_axis", 1))
    with Image.open(image_path) as original:
        original = original.convert("RGB")
    width, height = original.size
    if samples < 1 or raw["status"].shape != (height*samples, width*samples):
        raise ValueError("ray-grid dimensions/sampling do not match shadow.png")
    return metadata, raw, original, ray_path, samples


def generate(source, output, bounds=None, scale=4, rays=None):
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("diagnostic output must be separate from the input render directory")
    if not isinstance(scale, int) or scale < 1:
        raise ValueError("crop enlargement scale must be a positive integer")
    metadata, raw, original, ray_path, samples = load_saved_render(source, rays)
    arrays, details = extract(raw, metadata)
    metadata_path, image_path = source/"diagnostics.json", source/"shadow.png"
    width, height = original.size
    bounds = crop_bounds(bounds, width, height)
    geometry = metadata.get("geometry", {})
    if "inner_radius" in geometry and "outer_radius" in geometry:
        low, high = (float(geometry[key]/metadata["mass"]) for key in ("inner_radius", "outer_radius"))
        scale_source = "saved disk radial boundaries"
    elif arrays["radius_valid"].any():
        values = arrays["radius_m"][arrays["radius_valid"]]
        low, high = float(values.min()), float(values.max())
        if low == high:
            low, high = max(0, low*.95), high*1.05
        scale_source = "observed finite emitting radius range; disk boundaries not saved"
    else:
        low, high, scale_source = 0., 1., "placeholder; no emitting radius data available"
    if not 0 <= low < high:
        raise ValueError("invalid radius color scale in saved disk geometry")
    maps = colorize(arrays, low, high)
    categories = [int(v) for v in np.unique(arrays["crossing_number"][arrays["crossing_valid"]])]
    titles = dict(radius="First emitting radius r_emit / M", crossing="Ordinal disk-plane crossing at first emitting hit",
                  azimuth="First emitting azimuth in the disk plane")
    names = dict(radius="emission_radius", crossing="crossing_number", azimuth="emission_azimuth")
    output.mkdir(parents=True, exist_ok=True)
    crops = output/"crops"
    crops.mkdir(exist_ok=True)
    np.savez_compressed(output/"diagnostic_arrays.npz", **arrays, samples=samples,
                        crop_bounds=np.array(bounds), mass=metadata["mass"])
    x0, y0, x1, y1 = bounds
    ray_bounds = tuple(v*samples for v in bounds)
    crop_arrays = {name: value[y0*samples:y1*samples, x0*samples:x1*samples] for name, value in arrays.items()}
    np.savez_compressed(crops/"diagnostic_arrays.npz", **crop_arrays, samples=samples,
                        source_crop_bounds=np.array(bounds), mass=metadata["mass"])
    enlarged = ((x1-x0)*scale*samples, (y1-y0)*scale*samples)
    original_ray_size = original.resize((width*samples, height*samples), Image.Resampling.NEAREST)
    visible_crop = crop_raster(original_ray_size, bounds, samples, scale)
    visible_crop.save(crops/"rendered_pixels.png")
    subtitle = f"Source bounds [{x0},{y0},{x1},{y1}); {scale}x nearest-neighbor enlargement; no new rays"
    annotate(visible_crop, "Original visible-light image: shadow and branches", subtitle, "rendered", low, high, categories).save(crops/"rendered.png")
    crop_panels = [visible_crop]
    for kind, image in maps.items():
        image.save(output/f"{names[kind]}_pixels.png")
        annotate(image, titles[kind], f"Saved first valid opaque hits; {samples}x{samples} rays/source pixel", kind, low, high, categories).save(output/f"{names[kind]}.png")
        cropped = crop_raster(image, bounds, samples, scale)
        cropped.save(crops/f"{names[kind]}_pixels.png")
        annotate(cropped, titles[kind], subtitle, kind, low, high, categories).save(crops/f"{names[kind]}.png")
        crop_panels.append(cropped)
    # A compact comparison uses identical native pixel crops, only enlarged 2x.
    panel_size = ((x1-x0)*2*samples, (y1-y0)*2*samples)
    atlas = comparison_atlas(crop_panels, ("Visible image", "Emission radius", "Crossing number", "Emission azimuth"), panel_size)
    atlas.save(output/"crop_comparison.png")
    record = dict(input_render=str(source), source_rays=str(ray_path), geodesics_retraced=False,
                  source_sha256={path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in (metadata_path, ray_path, image_path)},
                  source_image_size=[width, height], ray_grid_shape=list(arrays["status"].shape), samples_per_axis=samples,
                  crop_bounds_xyxy_exclusive=list(bounds), crop_ray_bounds_xyxy_exclusive=list(ray_bounds),
                  enlargement=scale, effective_source_pixel_enlargement=scale*samples,
                  interpolation="nearest neighbor for every crop; no added angular resolution",
                  orientation="top-left origin; x=column increases right, y=row increases down; no flips",
                  panel_raster_offset_xy=[20, 55], unannotated_suffix="_pixels.png preserves the ray grid",
                  radius_scale_m=[low, high], radius_scale_source=scale_source, categories=categories, extraction=details,
                  azimuth_convention="degrees [0,360); projected global +X as zero (fallback +Y); e90=normal cross e0; right-handed about disk normal",
                  supersampling="Arrays and scientific maps retain every subray, never average angles/categories. Visible image is repeated onto that grid, not reconstructed per subray.",
                  color_maps=dict(radius="nine-anchor viridis approximation", crossing="categorical; legend lists each observed ordinal", azimuth="cyclic HSV hue; 0 and 360 match"))
    if metadata.get("spacetime",{}).get("type")=="kerr":
        record.update(spacetime=metadata["spacetime"],
                      radius_convention="Kerr r/M; not Euclidean radius",
                      azimuth_convention="outgoing Cartesian KS atan2(y,x), zero +X, positive toward +Y about +Z; [0,360) degrees; not BL azimuth")
    (output/"metadata.json").write_text(json.dumps(record, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("output/disk-edge-on-blackbody-hires"))
    parser.add_argument("--output", type=Path, default=Path("examples/disk_diagnostics"))
    parser.add_argument("--crop", type=int, nargs=4, metavar=("X0", "Y0", "X1", "Y1"))
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--rays", type=Path, help="override saved ray file for a moved/older companion render")
    args = parser.parse_args()
    try:
        record = generate(args.input, args.output, args.crop, args.scale, args.rays)
    except (ValueError, OSError, KeyError) as exc:
        parser.error(str(exc))
    print(f"Saved diagnostic maps and matching crops in {args.output}; no geodesics retraced")
    for warning in record["extraction"]["warnings"]:
        print("Diagnostic unavailable:", warning)


if __name__ == "__main__":
    main()
