"""Optional quadrature rendering and matched-background diagnostic output."""
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .background import linear_to_srgb, make_background
from .schwarzschild_orbit import ray_map


def shade(directions, status, background, samples=1):
    image = np.zeros(status.shape+(3,))
    mask = status == 1
    # Bound temporaries when sampling the large photographic texture.
    flat = np.flatnonzero(mask)
    pixels = image.reshape(-1, 3)
    rays = directions.reshape(-1, 3)
    for start in range(0, len(flat), 100000):
        ids = flat[start:start+100000]
        pixels[ids] = background.color(rays[ids])
    image[status == 2] = (1, 0, 0.7)
    if samples > 1:
        linear = np.where(image <= 0.04045, image/12.92, ((image+0.055)/1.055)**2.4)
        h, w = status.shape[0]//samples, status.shape[1]//samples
        image = linear_to_srgb(linear.reshape(h, samples, w, samples, 3).mean(axis=(1, 3)))
    return image


def save_image(path, image):
    Image.fromarray(np.uint8(np.round(np.clip(image, 0, 1)*255))).save(path)


def render_quadrature(metric, camera, integration, output, background, samples=1, workers=1, progress=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    execution = {}
    directions, status, count = ray_map(metric, camera, samples, workers=workers, execution=execution, progress=progress)
    if progress:
        progress(camera.height*samples, camera.height*samples)
    save_image(output/"shadow.png", shade(directions, status, background, samples))
    np.savez_compressed(output/"rays.npz", directions=directions, status=status, samples=samples)
    report = dict(method="Schwarzschild orbital quadrature", camera=asdict(camera),
                  mass=metric.mass, background=background.description,
                  samples_per_axis=samples, distinct_orbits=count,
                  counts={key: int(np.count_nonzero(status == code)) for code, key in
                          enumerate(("captured", "escaped", "unresolved"))},
                  diagnostic_scope="Counts are subpixel rays; no ODE null residual is inferred for quadrature rays.")
    if execution:
        report["execution"] = execution
    (output/"diagnostics.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    return report


def render_pair(metric, camera, output, options, samples=2):
    """Both backgrounds consume exactly the same saved subpixel ray map."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    directions, status, count = ray_map(metric, camera, samples)
    np.savez_compressed(output/"rays.npz", directions=directions, status=status, samples=samples)
    images = {}
    for mode in ("starfield", "grid"):
        background = make_background(dict(options, mode=mode))
        images[mode] = shade(directions, status, background, samples)
        save_image(output/f"{mode}.png", images[mode])
        h, w = camera.height*samples, camera.width*samples
        yy, xx = np.mgrid[:h, :w]
        f, r, u = camera.axes()
        scale = np.tan(np.deg2rad(camera.fov_degrees)/2)/w
        straight = f + scale*((2*xx+1-w)[..., None]*r+(h-2*yy-1)[..., None]*u)
        straight /= np.linalg.norm(straight, axis=-1, keepdims=True)
        save_image(output/f"unlensed_{mode}.png", shade(straight, np.ones((h, w), dtype=np.uint8), background, samples))
        if mode == "grid":
            # This labelled reference atlas is NOT composited onto the lensed view.
            lon, lat = np.meshgrid(np.linspace(np.pi, -np.pi, 1000), np.linspace(np.pi/2, -np.pi/2, 500))
            sky = np.stack([np.cos(lat)*np.cos(lon), np.cos(lat)*np.sin(lon), np.sin(lat)], axis=-1)
            atlas = Image.fromarray(np.uint8(background.color(sky @ background.rotation)*255))
            draw = ImageDraw.Draw(atlas)
            for degree in range(-150, 180, 30):
                draw.text(((180-degree)/360*1000+4, 250+4), f"{degree} deg", fill="white")
            for degree in range(-60, 90, 30):
                draw.text((505, (90-degree)/180*500+4), f"{degree} deg", fill="white")
            atlas.save(output/"grid_atlas.png")
        del background
    return directions, status, count, images
