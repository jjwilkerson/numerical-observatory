"""Assemble an optional disk scene; physical models stay separate from display."""
from dataclasses import dataclass, field

import numpy as np

from .display import DiskDisplay
from .emission import PowerLawBlackbody
from .emitter_motion import SchwarzschildCircular
from .surfaces import ThinAnnulus


@dataclass(frozen=True)
class DiskScene:
    surface: ThinAnnulus
    motion: SchwarzschildCircular
    emission: PowerLawBlackbody
    display: DiskDisplay


@dataclass(frozen=True)
class DiskConfig:
    enabled: bool = False
    inner_radius: float = 6.0
    outer_radius: float = 20.0
    inclination_degrees: float = 60.0
    position_angle_degrees: float = 0.0
    rotation: int = 1
    temperature_inner: float = 10000.0
    temperature_index: float = 0.75
    emissivity_scale: float = 1.0
    display: dict = field(default_factory=dict)


def make_disk(metric, options=None):
    config = DiskConfig(**(options or {}))
    if not isinstance(config.enabled, bool):
        raise ValueError("disk enabled must be a boolean")
    if not config.enabled:
        return None
    from .metric import Schwarzschild
    if type(metric) is not Schwarzschild:
        raise ValueError("luminous disk physics currently requires Schwarzschild; Kerr disks are not implemented")
    if not np.isfinite([config.inclination_degrees, config.position_angle_degrees]).all():
        raise ValueError("disk orientation must be finite")
    if config.inner_radius <= 3:
        raise ValueError("circular-orbit disk emission requires inner_radius>3M (stable orbits start at 6M)")
    i, pa = np.deg2rad([config.inclination_degrees, config.position_angle_degrees])
    normal = (np.sin(i)*np.sin(pa), np.sin(i)*np.cos(pa), np.cos(i))
    # This factory is explicitly Schwarzschild; generic surfaces/motion can be
    # assembled as DiskScene-compatible objects without modifying the camera.
    motion = SchwarzschildCircular(metric, normal, config.rotation)
    surface = ThinAnnulus(config.inner_radius*metric.mass, config.outer_radius*metric.mass, normal)
    return DiskScene(surface, motion,
                     PowerLawBlackbody(surface.inner_radius, config.temperature_inner,
                                       config.temperature_index, config.emissivity_scale),
                     DiskDisplay(**config.display))
