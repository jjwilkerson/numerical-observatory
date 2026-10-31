"""Pinhole camera in a local orthonormal frame for an event-based metric."""
from dataclasses import dataclass

import numpy as np

from .metric import ETA, Metric


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
    observer_frame: tuple | None = None
    observer: str = "static"

    def __post_init__(self):
        if (not isinstance(self.width, int) or not isinstance(self.height, int)
                or self.width < 1 or self.height < 1):
            raise ValueError("image dimensions must be positive integers")
        if not np.isfinite(self.fov_degrees) or not 0 < self.fov_degrees < 180:
            raise ValueError("horizontal FOV must lie between 0 and 180 degrees")
        if not np.isfinite(self.time):
            raise ValueError("camera time must be finite")
        if np.asarray(self.position).shape != (3,) or not np.isfinite(self.position).all():
            raise ValueError("position must be a finite three-vector")
        self.axes()
        if self.observer not in ("static", "zamo"):
            raise ValueError("observer must be static or zamo")

    @property
    def event(self):
        return np.r_[self.time, self.position]

    def axes(self):
        forward = unit(np.asarray(self.target) - np.asarray(self.position))
        right = unit(np.cross(forward, unit(self.up)))
        up = unit(np.cross(right, forward))
        return forward, right, up

    def tetrad(self, metric: Metric):
        """Rows are u,e_x,e_y,e_z. Default: observer at fixed coordinates.

        Symmetric orthonormalization of its rest-space metric preserves the
        previous Schwarzschild frame without assuming spherical symmetry.
        Supply a future-directed observer_frame for a moving observer or where
        the coordinate-time direction is not timelike (e.g. an ergoregion).
        """
        g = metric.covariant(self.event)
        if self.observer_frame is not None:
            frame = np.asarray(self.observer_frame, dtype=float)
            if (frame.shape != (4, 4) or not np.isfinite(frame).all()
                    or frame[0, 0] <= 0
                    or not np.allclose(frame @ g @ frame.T, ETA, atol=1e-10, rtol=1e-10)):
                raise ValueError("observer_frame must be an orthonormal tetrad")
            return frame
        if self.observer == "zamo":
            from .kerr import Kerr
            if type(metric) is Kerr:
                u = metric.zamo_velocity(self.event)
                # Project the Cartesian coordinate basis into the observer rest
                # space, then symmetrically orthonormalize its Gram matrix.
                ucov = g @ u
                projected = np.column_stack((np.zeros(3), np.eye(3))) + ucov[1:, None]*u
                gram = projected @ g @ projected.T
                values, vectors = np.linalg.eigh(gram)
                if np.any(values <= 0):
                    raise ValueError("ZAMO rest-space metric is not positive definite")
                frame = np.vstack((u, ((vectors/np.sqrt(values)) @ vectors.T) @ projected))
                return frame
            from .metric import Schwarzschild
            if type(metric) is not Schwarzschild:
                raise ValueError("ZAMO observers currently require Kerr or Schwarzschild")
        if g[0, 0] >= 0:
            raise ValueError("coordinate-static observer is not timelike; supply observer_frame")
        rest = g[1:, 1:] - np.outer(g[0, 1:], g[0, 1:])/g[0, 0]
        values, vectors = np.linalg.eigh(rest)
        if np.any(values <= 0):
            raise ValueError("observer rest-space metric must be positive definite")
        spatial = (vectors / np.sqrt(values)) @ vectors.T
        frame = np.zeros((4, 4))
        frame[0, 0] = 1/np.sqrt(-g[0, 0])
        frame[1:, 1:] = spatial
        frame[1:, 0] = -spatial @ g[0, 1:]/g[0, 0]
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
