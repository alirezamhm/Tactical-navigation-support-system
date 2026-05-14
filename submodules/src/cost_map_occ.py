import logging
import os
import pickle
import random
from typing import List, Tuple, Any, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches
from skimage import draw

from submodules.src.geometry.polygon import *
from submodules.src.utils.utils import scale_axis_labels, rotation_matrix
from submodules.src.occupancy_grid.occupancy_map import OccupancyGrid


# define an arbitrary max cost applied to a cell in the costmap
MAX_COST = 1e10


class CostMap_Occupancy:
    """
    Discretizes the environment into a 2D map and assigns a cost to each grid cell.
    Costmap for the predictive planner.
    This class serves more as a helper class for the predictive planner.
    """
    def __init__(self, cfg, scale: float, m: int, n: int, alpha: float = 10,
                 ship_mass: float = 1, horizon: int = None, margin: int = 1, channels: int = 1):
        """
        :param scale: the scaling factor for the costmap, divide by scale to get world units
        :param m: the height in world units of the channel
        :param n: the width in world units of the channel
        :param alpha: weight for the collision cost term
        :param ship_mass: mass of the ship in kg
        :param horizon: the horizon ahead of the ship that is considered for computing costs
        :param margin: the number of pixels to apply a max cost to at the boundaries of the channel
        """
        self.scale = scale  # scales everything by this factor
        if channels > 1:
            self.cost_map = np.zeros((int(m * scale), int(n * scale), channels))
        else:
            self.cost_map = np.zeros((int(m * scale), int(n * scale)))
        self.alpha = alpha
        self.ship_mass = ship_mass
        self.obstacles = []
        self.horizon = horizon * scale if horizon else None
        self.margin = margin

        self.logger = logging.getLogger(__name__)

        # apply a cost to the boundaries of the channel
        self.boundary_cost()

        self.occupancy = OccupancyGrid(grid_width=cfg.occ.grid_size, grid_height=cfg.occ.grid_size, map_width=cfg.occ.map_width, map_height=cfg.occ.map_height, ship_body=None)
        print("occ cost map dimention: ", self.occupancy.map_height, self.occupancy.map_width)
        self.occ_fig, self.occ_ax = plt.subplots(figsize=(10, 10))
        self.occ_ax.set_xlabel('')
        self.occ_ax.set_xticks([])
        self.occ_ax.set_ylabel('')
        self.occ_ax.set_yticks([])
        self.update_count = 0
        
    @property
    def shape(self):
        return self.cost_map.shape

    def boundary_cost(self) -> None:
        if not self.margin:
            return
        self.cost_map[:, :self.margin] = MAX_COST
        self.cost_map[:, -self.margin:] = MAX_COST

    def populate_costmap(self, centre, radius, pixels, normalization) -> None:
        rr, cc = pixels
        centre_x, centre_y = centre

        for (row, col) in zip(rr, cc):
            dist = np.sqrt((row - centre_y) ** 2 + (col - centre_x) ** 2)
            new_cost = max(0, (radius ** 2 - dist ** 2) / radius ** 2)
            old_cost = self.cost_map[row, col]
            self.cost_map[row, col] = min(MAX_COST, max(new_cost * normalization, old_cost))

        # make sure there are no pixels with 0 cost
        assert np.all(self.cost_map[rr, cc] > 0)


    def update(self, occ_map=None) -> None:
        """ Updates the costmap with the new obstacles and ship position and velocity 
        NOTE: same update function as above but for ROS
        """
        # clear costmap and obstacles
        self.cost_map[:] = 0
        self.cost_map[0:occ_map.shape[0]] = occ_map
        self.boundary_cost()

