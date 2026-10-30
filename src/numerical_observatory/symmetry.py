"""Exact rotational reuse for a centered Schwarzschild camera, without interpolation."""
from dataclasses import replace

import numpy as np

from .geodesic import trace
from .metric import Schwarzschild


class CenteredRayCache:
    """Equal image-plane radii share an orbit, rotated into each ray's plane.

    Every distinct radius is integrated with the full Hamiltonian ODE.
    Off-center views and other metrics take the direct tracing path.
    """

    def __init__(self, metric, camera, integration, enabled=True, domain=None):
        self.metric, self.camera, self.integration = metric, camera, integration
        self.domain = domain
        self.forward, self.right, self.up = camera.axes()
        radius = np.linalg.norm(camera.position)
        radial = np.asarray(camera.position)/radius if radius else np.zeros(3)
        self.enabled = bool(enabled and type(metric) is Schwarzschild and domain is None
                            and camera.observer_frame is None
                            and np.linalg.norm(self.forward+radial) < 1e-14)
        self.cache = {}
        self.integrated_rays = 0
        self.steps = self.evaluations = 0

    def ray(self, row, col, momentum):
        # Integer offsets identify equal radii exactly, including odd dimensions.
        ix = 2*col + 1 - self.camera.width
        iy = self.camera.height - 2*row - 1
        key = ix*ix + iy*iy
        if not self.enabled or key not in self.cache:
            if self.enabled:
                offset = np.sqrt(key)*np.tan(np.deg2rad(self.camera.fov_degrees)/2)/self.camera.width
                momentum = self.camera.momentum(self.metric, self.forward+offset*self.right)
            result = trace(self.metric, self.camera.event, momentum, self.integration, domain=self.domain)
            self.integrated_rays += 1
            self.steps += result.steps
            self.evaluations += result.evaluations
            if self.enabled:
                self.cache[key] = result
        else:
            result = self.cache[key]
        if self.enabled and result.direction is not None and key:
            transverse = (ix*self.right + iy*self.up)/np.sqrt(key)
            direction = (np.dot(result.direction, self.forward)*self.forward
                         + np.dot(result.direction, self.right)*transverse)
            result = replace(result, direction=direction/np.linalg.norm(direction))
        return result
