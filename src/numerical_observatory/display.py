"""Display transforms only; physical intensity and g are saved separately."""
from dataclasses import dataclass

import numpy as np
from scipy.constants import sigma, c

from .background import linear_to_srgb
from .emission import planck_nu
from .blackbody_color import blackbody_rgb


def srgb_to_linear(rgb):
    rgb = np.asarray(rgb)
    return np.where(rgb <= 0.04045, rgb/12.92, ((rgb+0.055)/1.055)**2.4)


@dataclass(frozen=True)
class DiskDisplay:
    exposure: float = 2.0
    reference_temperature: float = 10000.0
    tone_mapping: str = "reinhard"
    palette: str = "thermal"

    def __post_init__(self):
        if (not np.isfinite([self.exposure, self.reference_temperature]).all()
                or self.exposure <= 0 or self.reference_temperature <= 0
                or self.tone_mapping not in ("reinhard", "linear")
                or self.palette not in ("thermal", "gray", "blackbody")):
            raise ValueError("invalid disk display settings")

    def linear_rgb(self, intensity, observed_temperature):
        level = np.asarray(intensity)/(sigma/np.pi*self.reference_temperature**4)*self.exposure
        if self.tone_mapping == "reinhard":
            level = level/(1+level)
        if self.palette == "gray":
            color = np.ones(level.shape+(3,))
        elif self.palette == "blackbody":
            color = blackbody_rgb(observed_temperature)
        else:
            # Illustrative three-band color, NOT CIE or a calibrated camera RGB.
            color = planck_nu(c/np.array([650e-9, 550e-9, 450e-9]), np.asarray(observed_temperature)[..., None])
            color = color/np.max(color, axis=-1, keepdims=True)
        return np.clip(level[..., None]*color, 0, 1)
