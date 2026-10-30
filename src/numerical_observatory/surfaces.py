"""Opaque surface geometry and first-hit queries on dense geodesic solutions."""
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from scipy.optimize import brentq


@dataclass(frozen=True)
class SurfaceHit:
    affine: float
    event: np.ndarray
    momentum: np.ndarray
    crossing_index: int


class Surface(Protocol):
    def level(self, event): ...
    def contains(self, event) -> bool: ...
    def first_hit(self, solution) -> SurfaceHit | None: ...


class CoplanarRay(ValueError):
    """An exactly coplanar ray has no isolated thin-surface crossing."""


@dataclass(frozen=True)
class ThinAnnulus:
    inner_radius: float
    outer_radius: float
    normal: tuple = (0.0, 0.0, 1.0)

    def __post_init__(self):
        n = np.asarray(self.normal, dtype=float)
        if (not np.isfinite([self.inner_radius, self.outer_radius]).all()
                or not 0 < self.inner_radius < self.outer_radius
                or n.shape != (3,) or not np.isfinite(n).all() or np.linalg.norm(n) == 0):
            raise ValueError("annulus requires 0 < inner < outer and a finite nonzero normal")
        object.__setattr__(self, "normal", tuple(n/np.linalg.norm(n)))

    def level(self, event):
        return np.asarray(self.normal) @ np.asarray(event)[1:4]

    def contains(self, event):
        radius = np.linalg.norm(np.asarray(event)[1:4])
        return self.inner_radius <= radius <= self.outer_radius

    def crossings(self, solution):
        """All isolated plane crossings in affine order, including radial holes.

        Bracket on accepted ODE steps and their midpoints, then refine with the
        dense interpolant. Convergence with max_step/tolerances remains required:
        no sign-change method can promise arbitrarily many roots within a step.
        """
        times = np.unique(np.r_[solution.t, (solution.t[1:]+solution.t[:-1])/2])
        states = solution.sol(times)
        levels = np.asarray(self.normal) @ states[1:4]
        scale = max(1.0, float(np.max(np.linalg.norm(states[1:4], axis=0))))
        if np.max(np.abs(levels)) < 1e-12*scale:
            raise CoplanarRay("coplanar ray is ambiguous for an infinitesimally thin surface")
        roots = []
        for i in range(len(times)-1):
            if levels[i] == 0:
                roots.append(times[i])
            elif levels[i]*levels[i+1] < 0:
                roots.append(brentq(lambda t: self.level(solution.sol(t)[:4]),
                                     times[i], times[i+1], xtol=1e-11, rtol=1e-13))
        if levels[-1] == 0:
            roots.append(times[-1])
        hits = []
        for t in roots:
            if t <= solution.t[0]+1e-10 or (hits and abs(t-hits[-1].affine) < 1e-9):
                continue
            y = solution.sol(t)
            hits.append(SurfaceHit(float(t), y[:4].copy(), y[4:].copy(), len(hits)+1))
        return hits

    def first_hit(self, solution):
        return next((hit for hit in self.crossings(solution) if self.contains(hit.event)), None)
