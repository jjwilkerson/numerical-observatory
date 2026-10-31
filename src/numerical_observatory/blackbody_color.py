"""CIE 1931 blackbody chromaticity; no physical intensity or frequency shifts.

CMFs: CIE 2019, DOI 10.25039/CIE.DS.xvudnb9b, CC BY-SA 4.0.
See data/ATTRIBUTION.md. Integrate B_lambda over 360--830 nm, then retain
XYZ/sum(XYZ) only. Lazy lookup avoids spectral integration in pixel loops.
"""
from functools import lru_cache
from importlib.resources import files

import numpy as np
from scipy.constants import h, c, k


MIN_TEMPERATURE = 1000.0
MAX_TEMPERATURE = 1e7
TABLE_SIZE = 4097

# Matrix derived from IEC sRGB/BT.709 primary xy and D65 white xy (ICC registry).
# Columns of the forward matrix are primary XYZ, scaled so RGB=(1,1,1) is D65.
_primaries = np.array([[.64, .30, .15], [.33, .60, .06], [.03, .10, .79]])
_white = np.array([.3127/.3290, 1., (1-.3127-.3290)/.3290])
XYZ_TO_LINEAR_SRGB = np.linalg.inv(_primaries*np.linalg.solve(_primaries, _white))
XYZ_TO_LINEAR_SRGB.setflags(write=False)


def _integrate_xyz(temperatures):
    """Reference table builder: trapezoid quadrature of tabulated 1 nm CMFs."""
    with files("numerical_observatory").joinpath("data/CIE_xyz_1931_2deg.csv").open("r") as stream:
        data = np.loadtxt(stream, delimiter=",")
    wavelength = data[:, 0]*1e-9
    # 2hc² cancels in chromaticity. Scale each spectrum to prevent magnitude
    # from affecting integration conditioning; do not use its visible flux.
    xyz = []
    for chunk in np.array_split(np.asarray(temperatures), max(1, (len(temperatures)+255)//256)):
        spectrum = wavelength**-5/np.expm1((h*c/k)/(chunk[:, None]*wavelength))
        spectrum /= spectrum.max(axis=1, keepdims=True)
        # Explicit trapezoidal weights support NumPy >=1.24 (no trapezoid API).
        weights = np.ones(len(wavelength))
        weights[[0, -1]] = .5
        values = spectrum @ (data[:, 1:]*weights[:, None])
        xyz.append(values/values.sum(axis=1, keepdims=True))
    return np.concatenate(xyz)


@lru_cache(maxsize=1)
def _table():
    log_temperature = np.linspace(np.log(MIN_TEMPERATURE), np.log(MAX_TEMPERATURE), TABLE_SIZE)
    xyz = _integrate_xyz(np.exp(log_temperature))
    log_temperature.setflags(write=False)
    xyz.setflags(write=False)
    return log_temperature, xyz


def blackbody_xyz(temperature):
    """Return XYZ normalized to X+Y+Z=1, shape temperature.shape+(3,).

    Interpolate in log(T). Positive temperatures outside 1000--10^7 K use
    endpoint chromaticities; this never clamps the caller's physical radiance.
    Nonpositive/nonfinite temperatures are errors rather than arbitrary colors.
    """
    temperature = np.asarray(temperature, dtype=float)
    if not np.isfinite(temperature).all() or np.any(temperature <= 0):
        raise ValueError("blackbody display temperature must be finite and positive")
    knots, xyz = _table()
    return np.stack([np.interp(np.log(temperature), knots, xyz[:, channel])
                     for channel in range(3)], axis=-1)


def blackbody_rgb(temperature):
    """Linear sRGB chromaticity, negative gamut channels clipped, max channel=1.

    These are emissive XYZ values expressed using D65 sRGB primaries: there is
    no source-white adaptation and no D50 ICC PCS step. Normalize after gamut
    clipping to separate color from the existing bolometric display level.
    """
    rgb = np.maximum(blackbody_xyz(temperature) @ XYZ_TO_LINEAR_SRGB.T, 0)
    return rgb/np.max(rgb, axis=-1, keepdims=True)
