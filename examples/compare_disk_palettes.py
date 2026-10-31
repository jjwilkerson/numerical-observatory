"""Recolor saved rays only; never integrate geodesics or modify the source."""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from numerical_observatory.background import make_background, linear_to_srgb
from numerical_observatory.display import DiskDisplay, srgb_to_linear


def compare(source, output):
    source, output = Path(source), Path(output)
    if source.resolve() == output.resolve() or source.resolve() in output.resolve().parents:
        raise ValueError("comparison output must be separate from the source render directory")
    ray_file = source/"disk_rays.npz"
    checksum = hashlib.sha256(ray_file.read_bytes()).hexdigest()
    report = json.loads((source/"diagnostics.json").read_text(encoding="utf-8"))
    description = report["background"]
    grid = description["model"] == "celestial coordinate grid"
    keys = ("longitude_degrees", "latitude_degrees", "roll_degrees",
            "spacing_degrees", "line_width_degrees") if grid else (
            "longitude_degrees", "latitude_degrees", "roll_degrees", "exposure", "resolution")
    background = make_background(dict(mode="grid" if grid else "starfield",
                                      **{key: description[key] for key in keys if key in description}))
    with np.load(ray_file) as data:
        codes = data["status"]
        rgb = np.zeros(codes.shape+(3,))
        escaped, emitted = codes == 1, codes == 3
        rgb[escaped] = srgb_to_linear(background.color(data["directions"][escaped]))
        rgb[codes == 2] = (1, 0, .7)
        intensity = data["bolometric_intensity"][emitted]
        temperature = data["observed_temperature"][emitted]
        n = report["samples_per_axis"]
        height, width = codes.shape[0]//n, codes.shape[1]//n
        panels = []
        output.mkdir(parents=True, exist_ok=True)
        for palette in ("thermal", "blackbody"):
            display = replace(DiskDisplay(**report["display"]), palette=palette)
            rgb[emitted] = display.linear_rgb(intensity, temperature)
            pixels = linear_to_srgb(rgb.reshape(height, n, width, n, 3).mean(axis=(1, 3)))
            image = Image.fromarray(np.uint8(np.rint(255*pixels)))
            image.save(output/f"{palette}.png")
            panels.append(image)
    comparison = Image.new("RGB", (2*width, height+24), (18, 18, 18))
    draw = ImageDraw.Draw(comparison)
    for index, (palette, panel) in enumerate(zip(("thermal", "blackbody"), panels)):
        comparison.paste(panel, (index*width, 24))
        draw.text((index*width+8, 6), palette, fill="white")
    comparison.save(output/"comparison.png")
    assert hashlib.sha256(ray_file.read_bytes()).hexdigest() == checksum
    metadata = dict(source=str(source), source_rays_sha256=checksum, geodesics_retraced=False,
                    observed_temperature_range=[float(temperature.min()), float(temperature.max())],
                    display=report["display"], changed_parameter="palette only",
                    cie_data_license="CC BY-SA 4.0; see src/numerical_observatory/data/ATTRIBUTION.md")
    (output/"comparison.json").write_text(json.dumps(metadata, indent=2)+"\n", encoding="utf-8")
    print(f"Saved {output/'comparison.png'} from unchanged saved rays")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("examples/disks/edge_on_starfield"))
    parser.add_argument("--output", type=Path, default=Path("examples/blackbody_palette"))
    args = parser.parse_args()
    compare(args.source, args.output)
