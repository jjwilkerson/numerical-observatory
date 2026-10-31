"""Display-only color checks against an independent fine-grid integration."""
from importlib.resources import files
import hashlib
import tomllib

import numpy as np
import pytest
from scipy.constants import c, h, k, sigma
from scipy.integrate import simpson

from numerical_observatory.blackbody_color import (
    blackbody_xyz, blackbody_rgb, XYZ_TO_LINEAR_SRGB,
    MIN_TEMPERATURE, MAX_TEMPERATURE,
    _integrate_xyz,
)
from numerical_observatory.display import DiskDisplay
from numerical_observatory.emission import PowerLawBlackbody, planck_nu
from numerical_observatory.cli import load_configuration
from numerical_observatory.disk import make_disk


def reference_xyz(temperatures):
    """B_nu -> B_lambda Jacobian, SI constants, 0.25 nm Simpson quadrature.

    Independent from the runtime B_lambda/trapezoid/table implementation.
    CMFs are linearly interpolated as prescribed by CIE metadata.
    """
    resource = files("numerical_observatory").joinpath("data/CIE_xyz_1931_2deg.csv")
    with resource.open("r") as stream:
        data = np.loadtxt(stream, delimiter=",")
    nm = np.linspace(360, 830, 1881)
    wavelength = nm*1e-9
    frequency = c/wavelength
    cmf = np.array([np.interp(nm, data[:, 0], data[:, j]) for j in (1, 2, 3)]).T
    t = np.atleast_1d(temperatures)
    b_nu = 2*h*frequency**3/c**2/np.expm1(h*frequency/(k*t[:, None]))
    b_lambda = b_nu*c/wavelength**2
    xyz = simpson(b_lambda[..., None]*cmf, x=wavelength, axis=1)
    return xyz/xyz.sum(axis=1, keepdims=True)


def test_cie_dataset_integrity_and_license():
    data = files("numerical_observatory").joinpath("data/CIE_xyz_1931_2deg.csv").read_bytes()
    assert hashlib.sha256(data).hexdigest() == "fa663e3535a7e0763a745993a1f0a192eb0275ac46ad2d1befd7626841e713c1"
    assert "CC BY-SA 4.0" in files("numerical_observatory").joinpath("data/CIE_xyz_1931_2deg.csv_metadata.json").read_text()


@pytest.mark.parametrize("temperature", [3000., 5800., 10000., 15000.])
def test_representative_chromaticity_against_independent_spectrum(temperature):
    np.testing.assert_allclose(blackbody_xyz(temperature), reference_xyz(temperature)[0], atol=2e-6, rtol=0)


def test_lookup_accuracy_across_supported_range():
    temperatures = np.geomspace(MIN_TEMPERATURE, MAX_TEMPERATURE, 257)*1.00017
    temperatures = np.minimum(temperatures, MAX_TEMPERATURE)
    # Separate lookup interpolation error from the 1 nm spectral quadrature.
    np.testing.assert_allclose(blackbody_xyz(temperatures), _integrate_xyz(temperatures), atol=3e-7, rtol=0)
    np.testing.assert_allclose(blackbody_xyz(temperatures), reference_xyz(temperatures), atol=3e-6, rtol=0)


def test_white_point_and_hot_cold_colors():
    # sRGB's neutral white is D65, not equal-energy XYZ and not a Planck radiator.
    d65 = np.array([.3127/.329, 1., (1-.3127-.329)/.329])
    np.testing.assert_allclose(XYZ_TO_LINEAR_SRGB @ d65, 1, atol=1e-14)
    warm, solar, hot, hotter = blackbody_rgb([3000, 5800, 10000, 15000])
    assert warm[0] > warm[1] > warm[2]
    assert solar[0] > solar[1] > solar[2]
    assert hot[2] > hot[1] > hot[0]
    assert hotter[0]/hotter[2] < hot[0]/hot[2]
    assert not np.allclose(blackbody_rgb(6504), 1, atol=1e-3)


def test_smooth_finite_nonnegative_colors_and_endpoint_policy():
    t = np.geomspace(MIN_TEMPERATURE, MAX_TEMPERATURE, 20000)
    rgb = blackbody_rgb(t)
    assert np.isfinite(rgb).all() and (rgb >= 0).all() and (rgb <= 1).all()
    np.testing.assert_allclose(rgb.max(axis=1), 1, atol=1e-14)
    assert np.max(np.abs(np.diff(rgb, axis=0))) < .002
    np.testing.assert_array_equal(blackbody_rgb(10), blackbody_rgb(MIN_TEMPERATURE))
    np.testing.assert_array_equal(blackbody_rgb(1e10), blackbody_rgb(MAX_TEMPERATURE))
    for temperature in (0, -1, np.nan, np.inf):
        with pytest.raises(ValueError, match="temperature"):
            blackbody_rgb(temperature)
    intensity = np.geomspace(1e-20, 1e30, len(t))
    for mapping in ("linear", "reinhard"):
        result = DiskDisplay(palette="blackbody", tone_mapping=mapping).linear_rgb(intensity, t)
        assert np.isfinite(result).all() and (result >= 0).all() and (result <= 1).all()


def test_shifted_temperatures_and_bolometric_brightness_are_separate():
    emission = PowerLawBlackbody(6, temperature_inner=10000)
    event = np.array([0., 6., 0., 0.])
    shifts = np.array([.3, .58, 1., 1.5])
    temperature = shifts*emission.temperature(event)
    physical = emission.observed_bolometric(event, shifts)
    before = physical.copy()
    display = DiskDisplay(palette="blackbody")
    result = display.linear_rgb(physical, temperature)
    level = physical/(sigma/np.pi*display.reference_temperature**4)*display.exposure
    level /= 1+level
    np.testing.assert_allclose(result, level[:, None]*blackbody_rgb([3000, 5800, 10000, 15000]), rtol=1e-14)
    np.testing.assert_allclose(result.max(axis=1), level, rtol=1e-14)
    np.testing.assert_array_equal(physical, before)
    np.testing.assert_allclose(physical/emission.bolometric(event), shifts**4, rtol=1e-14)


@pytest.mark.parametrize("mapping", ["linear", "reinhard"])
def test_original_palettes_unchanged(mapping):
    intensity = np.array([0., 1e5, 1e8])
    t = np.array([3000., 5800., 15000.])
    level = intensity/(sigma/np.pi*10000**4)*2
    if mapping == "reinhard":
        level = level/(1+level)
    three_bands = planck_nu(c/np.array([650e-9, 550e-9, 450e-9]), t[:, None])
    three_bands /= three_bands.max(axis=-1, keepdims=True)
    for palette, color in (("gray", np.ones((3, 3))), ("thermal", three_bands)):
        np.testing.assert_array_equal(DiskDisplay(palette=palette, tone_mapping=mapping).linear_rgb(intensity, t),
                                      np.clip(level[:, None]*color, 0, 1))
    assert DiskDisplay().palette == "thermal"


def test_blackbody_configuration_and_invalid_palette(tmp_path):
    path = tmp_path/"blackbody.toml"
    path.write_text('[disk]\nenabled = true\n[disk.display]\npalette = "blackbody"\n', encoding="utf-8")
    metric, _, _ = load_configuration(path)
    options = tomllib.loads(path.read_text(encoding="utf-8"))["disk"]
    assert make_disk(metric, options).display.palette == "blackbody"
    options["display"]["palette"] = "orange"
    with pytest.raises(ValueError, match="display"):
        make_disk(metric, options)
