"""Pinhole camera and a static Schwarzschild observer's orthonormal frame."""
from dataclasses import dataclass

import numpy as np

from .metric import Schwarzschild


def unit(vector):
    vector = np.asarray(vector, dtype=float)
    length = np.linalg.norm(vector)
    if vector.shape != (3,) or not np.isfinite(vector).all() or length < 1e-14:
        raise ValueError("expected a finite, nonzero three-vector")
    return vector / length


@dataclass(frozen=True)
class Camera:
    position: tuple = (0.0, 0.0, 20.0)
    target: tuple = (0.0, 0.0, 0.0)
    up: tuple = (0.0, 1.0, 0.0)
    fov_degrees: float = 60.0  # horizontal field of view
    width: int = 320
    height: int = 240
    time: float = 0.0

    def __post_init__(self):
        if (not isinstance(self.width, int) or not isinstance(self.height, int)
                or self.width < 1 or self.height < 1):
            raise ValueError("image dimensions must be positive integers")
        if not np.isfinite(self.fov_degrees) or not 0 < self.fov_degrees < 180:
            raise ValueError("horizontal FOV must lie between 0 and 180 degrees")
        if not np.isfinite(self.time):
            raise ValueError("camera time must be finite")
        unit(self.position)
        self.axes()

    @property
    def event(self):
        return np.r_[self.time, self.position]

    def axes(self):
        forward = unit(np.asarray(self.target) - np.asarray(self.position))
        right = unit(np.cross(forward, unit(self.up)))
        up = unit(np.cross(right, forward))
        return forward, right, up

    def tetrad(self, metric: Schwarzschild):
        """Rows are u, e_x, e_y, e_z, spatial axes anchored to Cartesian axes.

        Radial spatial axis has time component -f/sqrt(1-f). Tangential
        axes are unchanged. This map is an isometry from local Euclidean 3-space.
        """
        r = np.linalg.norm(self.position)
        f = 2 * metric.mass / r
        if f >= 1:
            raise ValueError("a static camera must be outside r=2M")
        n = np.asarray(self.position) / r
        a = np.sqrt(1 - f)
        frame = np.zeros((4, 4))
        frame[0, 0] = 1 / a
        frame[1:, 0] = -f / a * n
        frame[1:, 1:] = np.eye(3) + (a - 1) * np.outer(n, n)
        return frame

    def momentum(self, metric, direction):
        """Past-directed null covector, unit local energy; direction points on sky."""
        tangent = np.r_[-1.0, unit(direction)] @ self.tetrad(metric)
        return metric.covariant(self.event) @ tangent

    def rays(self, metric):
        forward, right, up = self.axes()
        half_width = np.tan(np.deg2rad(self.fov_degrees) / 2)
        for row in range(self.height):
            y = (1 - 2 * (row + 0.5) / self.height) * half_width * self.height / self.width
            for col in range(self.width):
                x = (2 * (col + 0.5) / self.width - 1) * half_width
                yield row, col, self.momentum(metric, forward + x * right + y * up)
