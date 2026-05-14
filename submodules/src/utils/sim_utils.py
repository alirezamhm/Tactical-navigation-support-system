from typing import List

import numpy as np
import pymunk

from submodules.src.ship import Ship


ICE_THICKNESS_DEFAULT = 1
ICE_ELASTICITY = 0.01
ICE_FRICTION = 1.0
CURRENT_MAX = 1
ICE_DENSITY = 0.001

def create_polygon(space, vertices, x, y, density, thickness, initial_velocity=None):
    body = pymunk.Body(body_type=pymunk.Body.DYNAMIC)
    body.position = (x, y)
    
    dummy_shape = pymunk.Poly(None, vertices)
    centre_of_g = dummy_shape.center_of_gravity
    vs = [(x - centre_of_g[0], y - centre_of_g[1]) for x, y in vertices]
    shape = pymunk.Poly(body, vs, radius=0.02)
    
    shape.density = density * thickness
    shape.elasticity = ICE_ELASTICITY
    shape.friction = ICE_FRICTION
    
    space.add(body, shape)
    return shape


def generate_sim_obs(space, obstacles: List[dict], density):
    return [
        create_polygon(
            space, (obs['vertices'] - np.array(obs['centre'])).tolist(),
            *obs['centre'],
            density=density,
            thickness=obs['thickness'] if 'thickness' in obs else ICE_THICKNESS_DEFAULT
        )
        for obs in obstacles
    ]


class WavyPolynomial:
    def __init__(self, coeffs, x_min, x_max, y_min, y_max, width, speed, n_samples=2000):
        self.coeffs = np.asarray(coeffs)
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max

        self.x_center = 0.5 * (x_min + x_max)
        self.A = 0.9 * (x_max - x_min) / 2
        
        self.width = width
        self.speed = speed
        
        t = np.linspace(0.0, 1.0, n_samples)
        q = np.zeros_like(t)
        for k, a in enumerate(self.coeffs):
            q += a * t ** k
        self.q_max = np.max(np.abs(q))
        if self.q_max == 0:
            self.q_max = 1.0  # avoid division by zero

    def __call__(self, y):
        y = np.asarray(y)
        t = (y - self.y_min) / (self.y_max - self.y_min)

        q = np.zeros_like(t, dtype=float)
        for k, a in enumerate(self.coeffs):
            q += a * t ** k

        q /= self.q_max

        return self.x_center + self.A * q

def apply_currents(polygons: List[pymunk.Poly], currents: WavyPolynomial) -> None:
    """
    Apply ocean current to ice floes using a wavy polynomial current pattern.

    Parameters:
        polygons (List[pymunk.Poly]): List of polygon shapes representing ice floes.
        currents (WavyPolynomial | None): A WavyPolynomial object defining the current pattern
    """
    if currents is None:
        return
    
    for poly in polygons:
        poly_center = poly.body.position
        poly_x, poly_y = poly_center.x, poly_center.y
        
        # Check if polygon's y-coordinate is within the current's y-range
        if currents.y_min <= poly_y <= currents.y_max:
            # Get the x-coordinate on the wave at this y position
            wave_x = currents(poly_y)
            
            # Check if polygon is within the current width
            half_width = currents.width / 2
            if wave_x - half_width <= poly_x <= wave_x + half_width:
                poly.body.apply_impulse_at_local_point(
                    impulse=pymunk.Vec2d(currents.speed, 0) * poly.mass,
                    point=(0, 0)
                )
    
