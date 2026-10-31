"""Optional NEW tracing of a source-image window at denser angular sampling.

Never launched by disk_diagnostics. Uses the existing serial ODE renderer with
its exact numerical settings. Window rays preserve the original pinhole camera.
"""
import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np

from .background import make_background
from .camera import Camera
from .disk import DiskScene
from .disk_diagnostics import crop_bounds
from .display import DiskDisplay
from .emission import PowerLawBlackbody
from .emitter_motion import SchwarzschildCircular
from .geodesic import Integration
from .metric import Schwarzschild
from .render import render
from .surfaces import ThinAnnulus


@dataclass(frozen=True)
class WindowCamera(Camera):
    reference_width: int = 640
    reference_height: int = 480
    origin_x: int = 0
    origin_y: int = 0

    def rays(self, metric):
        forward, right, up = self.axes()
        half_width = np.tan(np.deg2rad(self.fov_degrees)/2)
        for row in range(self.height):
            global_row = self.origin_y+row
            y = (1-2*(global_row+.5)/self.reference_height)*half_width*self.reference_height/self.reference_width
            for col in range(self.width):
                global_col = self.origin_x+col
                x = (2*(global_col+.5)/self.reference_width-1)*half_width
                yield row, col, self.momentum(metric, forward+x*right+y*up)


def window_camera(camera, bounds, density):
    if not isinstance(density, int) or density < 1:
        raise ValueError("density must be a positive integer")
    x0, y0, x1, y1 = crop_bounds(bounds, camera.width, camera.height)
    return WindowCamera(**(asdict(camera) | dict(width=(x1-x0)*density, height=(y1-y0)*density)),
                        reference_width=camera.width*density, reference_height=camera.height*density,
                        origin_x=x0*density, origin_y=y0*density)


def trace_window(source, output, bounds, density):
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("new tracing output must be separate from the source")
    report = json.loads((source/"diagnostics.json").read_text(encoding="utf-8"))
    metric = Schwarzschild(report["mass"])
    camera = Camera(**report["camera"])
    bounds = crop_bounds(bounds, camera.width, camera.height)
    cropped = window_camera(camera, bounds, density)
    scene = DiskScene(ThinAnnulus(**report["geometry"]),
                      SchwarzschildCircular(metric, report["geometry"]["normal"], report["rotation"]),
                      PowerLawBlackbody(**report["emission"]), DiskDisplay(**report["display"]))
    sky = report["background"]
    grid = sky["model"] == "celestial coordinate grid"
    keys = ("longitude_degrees", "latitude_degrees", "roll_degrees", "spacing_degrees", "line_width_degrees") if grid else (
        "longitude_degrees", "latitude_degrees", "roll_degrees", "exposure", "resolution")
    background = make_background(dict(mode="grid" if grid else "starfield", **{key: sky[key] for key in keys if key in sky}))
    # Explicitly serial with symmetry disabled: the generic centered-orbit
    # optimizations and multiprocessing pixel generator assume a full frame.
    result = render(metric, cropped, Integration(**report["integration"]), output,
                    disk=scene, background=background, method="ode", workers=1, samples=1,
                    use_symmetry=False,
                    progress=lambda row, total: print(f"Window rows {row}/{total}", flush=True) if row % max(1, total//10) == 0 else None)
    result["diagnostic_window"] = dict(source_render=str(source), source_camera=report["camera"],
                                     crop_bounds_xyxy_exclusive=list(bounds), density=density,
                                     global_virtual_image_size=[cropped.reference_width, cropped.reference_height],
                                     virtual_pixel_origin_xy=[cropped.origin_x, cropped.origin_y],
                                     note="NEW geodesics; exact full-camera subpixel positions, not a narrower centered FOV")
    (output/"diagnostics.json").write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--crop", type=int, nargs=4, required=True, metavar=("X0", "Y0", "X1", "Y1"))
    parser.add_argument("--density", type=int, default=2)
    args = parser.parse_args()
    try:
        trace_window(args.input, args.output, args.crop, args.density)
    except (ValueError, KeyError, OSError) as exc:
        parser.error(str(exc))
