"""Catalog-based astronomical sky texture; no disk or artificial ring emission."""
from importlib.resources import files

import numpy as np
from PIL import Image


def sky_rotation(longitude_degrees, latitude_degrees, roll_degrees):
    lon, lat, roll = np.deg2rad([longitude_degrees, latitude_degrees, roll_degrees])
    c, s = np.cos(roll), np.sin(roll)
    base = np.array([[0, 0, -1], [-c, -s, 0], [-s, c, 0]])
    c, s = np.cos(lat), np.sin(lat)
    tilt = np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]])
    c, s = np.cos(lon), np.sin(lon)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]) @ tilt @ base


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
        self.rotation = sky_rotation(*values[:3])
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


class CelestialGrid:
    """Analytic sky grid: cool meridians, warm parallels, 5-degree minor lines.

    Coordinates live on the celestial sphere, never on the image plane.
    Indexed line colors and cell shading expose parity and multiple images.
    Subpixel lines still require adequate ray sampling; no artificial widening
    of lines near a caustic or the shadow is applied.
    """
    def __init__(self, longitude_degrees=0.0, latitude_degrees=10.0,
                 roll_degrees=25.0, spacing_degrees=15.0, line_width_degrees=0.25):
        if (not np.isfinite([longitude_degrees, latitude_degrees, roll_degrees,
                             spacing_degrees, line_width_degrees]).all()
                or spacing_degrees <= 0 or line_width_degrees <= 0
                or not np.isclose(180/spacing_degrees, round(180/spacing_degrees))):
            raise ValueError("grid spacing must divide 180 degrees and widths must be positive")
        self.rotation = sky_rotation(longitude_degrees, latitude_degrees, roll_degrees)
        self.spacing = np.deg2rad(spacing_degrees)
        self.width = np.deg2rad(line_width_degrees)
        self.description = dict(model="celestial coordinate grid", longitude_degrees=longitude_degrees,
                                latitude_degrees=latitude_degrees, roll_degrees=roll_degrees,
                                spacing_degrees=spacing_degrees, line_width_degrees=line_width_degrees,
                                legend="cool cyan/blue meridians; warm red/gold parallels; gray minor lines at spacing/3",
                                accretion_disk=False)

    def color(self, direction):
        sky = np.asarray(direction) @ self.rotation.T
        lon = np.arctan2(sky[..., 1], sky[..., 0])
        lat = np.arcsin(np.clip(sky[..., 2], -1, 1))
        ilon, ilat = np.rint(lon/self.spacing), np.rint(lat/self.spacing)
        dl = np.arcsin(np.clip(np.abs(np.cos(lat)*np.sin(lon-ilon*self.spacing)), 0, 1))
        db = np.abs(lat-ilat*self.spacing)
        # Smooth, fixed angular width. Work in linear RGB for supersampling.
        longitude_line = np.exp(-(dl/self.width)**4)[..., None]
        latitude_line = np.exp(-(db/self.width)**4)[..., None]
        cell = (np.floor(lon/self.spacing)+np.floor(lat/self.spacing)) % 2
        base = np.stack([0.012+0.009*cell, 0.018+0.012*cell, 0.03+0.02*cell], axis=-1)
        minor = self.spacing/3
        minor_dist = np.minimum(np.abs(lat-np.rint(lat/minor)*minor),
                               np.abs(np.cos(lat)*np.sin(lon-np.rint(lon/minor)*minor)))
        base += 0.08*np.exp(-(minor_dist/(self.width*0.45))**4)[..., None]
        phase = ilon*self.spacing
        cool = np.stack([0.06+0*phase, 0.5+0.35*np.cos(phase), 0.7+0.25*np.sin(phase)], axis=-1)
        warm = np.stack([0.95+0*ilat, 0.25+0.45*(ilat % 3)/2, 0.08+0.2*(ilat % 2)], axis=-1)
        color = (1-longitude_line)*base+longitude_line*cool
        color = (1-latitude_line)*color+latitude_line*warm
        color += longitude_line*latitude_line*0.35
        return linear_to_srgb(color)


def make_background(options=None):
    options = dict(options or {})
    mode = options.pop("mode", "starfield")
    if mode == "starfield":
        for key in ("spacing_degrees", "line_width_degrees"):
            options.pop(key, None)
        return Starfield(**options)
    if mode == "grid":
        for key in ("resolution", "exposure"):
            options.pop(key, None)
        return CelestialGrid(**options)
    raise ValueError("background mode must be 'starfield' or 'grid'")
