"""Local rest-frame emission and invariant relativistic intensity transfer."""
from dataclasses import dataclass, field

import numpy as np
from scipy.constants import sigma, h, c, k


def photon_frequency(momentum, velocity, past_directed=True):
    """Frequency up to h: physical future p gives -p.u; backward p gives +p.u.

    Do not take abs: a wrong momentum/time-orientation convention must fail.
    """
    value = float(np.dot(momentum, velocity))*(1 if past_directed else -1)
    if not np.isfinite(value) or value <= 0:
        raise ValueError("photon and observer time orientations are inconsistent")
    return value


def frequency_shift(observer_momentum, observer_velocity, emitter_momentum,
                    emitter_velocity, past_directed=True):
    return (photon_frequency(observer_momentum, observer_velocity, past_directed)
            / photon_frequency(emitter_momentum, emitter_velocity, past_directed))


def planck_nu(frequency, temperature):
    frequency = np.asarray(frequency, dtype=float)
    return 2*h*frequency**3/c**2/np.expm1(np.clip(h*frequency/(k*temperature), 0, 700))


@dataclass(frozen=True)
class PowerLawBlackbody:
    inner_radius: float
    temperature_inner: float = 10000.0
    temperature_index: float = 0.75
    emissivity_scale: float = 1.0

    def __post_init__(self):
        if (not np.isfinite(list(vars(self).values())).all() or self.inner_radius <= 0
                or self.temperature_inner <= 0 or self.temperature_index < 0
                or self.emissivity_scale < 0):
            raise ValueError("invalid blackbody emission parameters")

    def temperature(self, event):
        return self.temperature_inner*(np.linalg.norm(event[1:4])/self.inner_radius)**(-self.temperature_index)

    def bolometric(self, event):
        # Isotropic radiance of each face (flux would be pi times this).
        return self.emissivity_scale*sigma/np.pi*self.temperature(event)**4

    def observed_bolometric(self, event, shift):
        # I_nu_obs(nu)=g^3 I_nu_emit(nu/g); integrating dnu adds another g.
        return shift**4*self.bolometric(event)

    def observed_specific(self, event, shift, frequency):
        return self.emissivity_scale*shift**3*planck_nu(frequency/shift, self.temperature(event))


@dataclass(frozen=True)
class KerrPowerLawBlackbody(PowerLawBlackbody):
    """The same toy emission law evaluated in Kerr radial coordinate r."""
    # keyword-only keeps the existing physical parameter ordering.
    metric: object = field(kw_only=True)

    def __post_init__(self):
        from .kerr import Kerr
        if type(self.metric) is not Kerr:
            raise ValueError('Kerr radial emission requires the exact Kerr metric')
        PowerLawBlackbody(self.inner_radius,self.temperature_inner,
                          self.temperature_index,self.emissivity_scale)

    def temperature(self,event):
        return self.temperature_inner*(self.metric.radius(event)/self.inner_radius)**(-self.temperature_index)
