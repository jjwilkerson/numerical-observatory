"""Catalog-based astronomical sky texture; no disk or artificial ring emission."""
from importlib.resources import files

import numpy as np
from PIL import Image


def linear_to_srgb(rgb):
    rgb = np.clip(rgb, 0, 1)
    return np.where(rgb <= 0.0031308, 12.92*rgb, 1.055*rgb**(1/2.4)-0.055)


class Starfield:
    """NASA Deep Star Maps (2012), galactic plate-carree, bilinear sampling.

    Decode sRGB before interpolation/exposure, then encode for display.
    This catalog visualization is not calibrated photometry. Sky orientation
    is independent of the camera and black hole. Longitude wraps at the seam.
    """

    def __init__(self, longitude_degrees=0.0, latitude_degrees=10.0,
                 roll_degrees=25.0, exposure=3.0, resolution=4):
        values = (longitude_degrees, latitude_degrees, roll_degrees, exposure)
        if not np.isfinite(values).all() or exposure <= 0:
            raise ValueError("sky angles must be finite and exposure must be positive")
        if resolution not in (4, 8):
            raise ValueError("star-map resolution must be 4 or 8 (kilopixels)")
        with files("numerical_observatory").joinpath(f"data/starmap_g{resolution}k.jpg").open("rb") as handle:
            with Image.open(handle) as image:
                rgb = np.asarray(image.convert("RGB"), dtype=np.float32)/255
        self.texture = np.where(rgb <= 0.04045, rgb/12.92, ((rgb+0.055)/1.055)**2.4)
        self.exposure = exposure
        lon, lat, roll = np.deg2rad(values[:3])
        c, s = np.cos(roll), np.sin(roll)
        base = np.array([[0, 0, -1], [-c, -s, 0], [-s, c, 0]])
        c, s = np.cos(lat), np.sin(lat)
        tilt = np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]])
        c, s = np.cos(lon), np.sin(lon)
        turn = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        self.rotation = turn @ tilt @ base
        self.description = {
            "model": f"NASA catalog-based Deep Star Maps (2012), galactic {resolution}k",
            "source": "https://svs.gsfc.nasa.gov/3895/",
            "credit": "NASA/Goddard Space Flight Center Scientific Visualization Studio",
            "longitude_degrees": longitude_degrees, "latitude_degrees": latitude_degrees,
            "roll_degrees": roll_degrees, "exposure": exposure, "accretion_disk": False,
            "resolution": resolution,
        }

    def color(self, direction):
        """Return display sRGB for one direction or an array (..., 3)."""
        sky = np.asarray(direction) @ self.rotation.T
        lon = np.arctan2(sky[..., 1], sky[..., 0])
        lat = np.arcsin(np.clip(sky[..., 2], -1, 1))
        height, width = self.texture.shape[:2]
        x = (0.5-lon/(2*np.pi))*width - 0.5
        y = np.clip((0.5-lat/np.pi)*height - 0.5, 0, height-1)
        ix, iy = np.floor(x).astype(int), np.floor(y).astype(int)
        fx, fy = (x-ix)[..., None], (y-iy)[..., None]
        x0, x1 = ix % width, (ix+1) % width
        y1 = np.minimum(iy+1, height-1)
        upper = (1-fx)*self.texture[iy, x0] + fx*self.texture[iy, x1]
        lower = (1-fx)*self.texture[y1, x0] + fx*self.texture[y1, x1]
        return linear_to_srgb(self.exposure*((1-fy)*upper + fy*lower))
