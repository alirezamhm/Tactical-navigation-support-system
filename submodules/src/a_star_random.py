""" A* search algorithm for finding a path by searching a graph of nodes connected by primitives """
import copy
import logging
import queue
from typing import Tuple, Union

import numba
import numpy as np
from numba import jit  # speeds up some computations

from submodules.src.cost_map import CostMap
from submodules.src.dubins_helpers.heuristic import dubins_heuristic
from submodules.src.path_smoothing import path_smoothing
from submodules.src.primitives import Primitives
from submodules.src.ship import Ship
from submodules.src.swath import Swath, rotate_swath, compute_swath_cost, view_swath
from submodules.src.utils.hashmap import HashMap
from submodules.src.utils.priority_queue import PriorityQueue
from submodules.src.utils.utils import heading_to_world_frame, rotation_matrix, M__2_PI
from matplotlib import pyplot as plt


class AStarRandom:
    def __init__(self, weight: float, cmap: CostMap, prim: Primitives, ship: Ship,
                 swath_dict: Swath, swath_dict_no_padding: Swath, ship_no_padding: Ship, use_ice_model: bool = False, smooth_path: Union[bool, dict] = False, **kwargs):
        self.weight = weight  # static weighting for heuristic
        self.smooth_path = smooth_path  # False is disabled, otherwise dict for smooth path params
        self.cmap = cmap
        self.prim = prim
        self.ship = ship
        self.max_val = int(self.prim.max_prim + self.ship.max_ship_length // 2)
        self.orig_swath_dict = swath_dict
        self.orig_swath_dict_no_pad = swath_dict_no_padding
        self.ship_no_pad = ship_no_padding
        self.max_val_no_pad = int(self.prim.max_prim + self.ship_no_pad.max_ship_length // 2)
        self.logger = logging.getLogger(__name__)

        # initialize member vars that are updated and used during planning
        self.cost_map = None
        self.swath_dict = None
        self.swath_arg_dict = None
        self.rotated_prims = None
        # this specifies how much below ship and above goal to include as part of costmap subset
        self.margin = kwargs.get('margin', int(5 * self.cmap.scale))

        # variable used to diagnose planning failures
        self.diagnostics: dict = None

        self.h_baseline = kwargs.get('h_baseline', False)

        if self.smooth_path:
            self.smooth_path_kwargs = dict(step_size=self.prim.step_size,
                                           ship_vertices=self.ship.vertices,
                                           turning_radius=self.prim.turning_radius,
                                           **self.smooth_path)

        # the way we represent nodes on lattice is odd since we use global coordinates
        # for x and y while for heading we use the lattice heading units

        self.planing_instance = 0
        print("Initialzed A Star Search!")

    
    
    # @profile  # to profile code run `python -m memory_profiler`
    def search(self, start: tuple, goal_y: float, 
               occ_map=None, centroids=None, footprint=None, ship_vertices=None, use_ice_model=False, debug=False, 
               prediction_horizon=None, goal_pos=None, goal_dis=None):
        """
        :param start: tuple of the form (x, y, theta) where x and y are in global coordinates
        :param goal_y: goal y position in global coordinates
        :param occ_map: initial ice concentration observation. If None, then do not use ice motion estimator
        :param centroids: initial ice centroids observation. If None, then do not use ice motion estimator
        :param footprint: initial footprint observation. If None, then do not use ice motion estimator
        :param ship_vertices: unscaled and unpadded original vertices for constructing footprint
        :param debug: whether to visualize ice model inputs
        :param prediction_horizon: if provided, then the range in which occupancy prediction will take place
        :param goal_pos: if provided, this is treated as a low-level planner which plans toward a goal position, instead of a line
        :param goal_dis: distance threshold to the goal_pos
        """
        # print("START PLANNING INSTANCE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~: ", self.planing_instance, "\n")
        # initialize memos

        self.diagnostics = None

        self.swath_dict = {}
        self.swath_arg_dict = {}
        self.swath_dict_no_pad = {}
        self.swath_arg_dict_no_pad = {}
        theta_0 = start[2] % M__2_PI
        R = rotation_matrix(theta_0)
        # reset rotated primitives
        self.rotated_prims = {}

        if prediction_horizon is None:
            prediction_horizon = float('inf')

        if goal_pos is not None:
            goal_y = goal_pos[1]

        # get subset of costmap based on finite horizon
        lower_lim = max(0, int(start[1]) - self.margin)
        upper_lim = min(self.cmap.shape[0], int(goal_y) + self.margin)
        self.cost_map = self.cmap.cost_map[lower_lim: upper_lim]

        # transform start and goal nodes
        start = (start[0], start[1] - lower_lim, 0)
        goal_y = goal_y - lower_lim
        if goal_pos is not None:
            goal_pos[1] = goal_pos[1] - lower_lim

        # use custom spatial hash maps since we are using the finite precision
        # coordinates of the nodes as the keys to the map

        # dicts to keep track of all the relevant path information
        # these maps are used to build the optimal path at the end of search
        came_from = {start: None}
        came_from_by_edge = {start: None}
        g_score = {start: 0}
        path_length = {start: 0}

        iter = 0
        max_iter = 1000
        node = start
        while iter < max_iter:
            # compute dis to goal node
            if goal_pos is not None:
                dis_to_goal = ((node[0] - goal_pos[0])**2 + (node[1] - goal_pos[1])**2)**(0.5)

            # high-level planner, passing the goal line
            # low-level planner, two possibilities: 1. get close enough to the target;  2. passing the environment final goal line
            if ((goal_pos is None) and node[1] >= goal_y) or \
            ((goal_pos is not None) and (dis_to_goal <= goal_dis or (node[1] + lower_lim) >= self.cmap.occupancy.occ_map_height)):
                print('Found path! node {} goal {}'.format(node, goal_y))

                # build path goal ~> start
                goal = node
                g_score[goal] = 0
                node_path = [node]
                node_path_length = [path_length[node]]

                while node != start:
                    pred = came_from[node]
                    node = pred
                    node_path.append(node)
                    node_path_length.append(path_length[node])

                node_path.reverse()  # we want start ~> goal

                full_path, full_swath, prim_count, edge_seq = self.build_path(node_path, came_from_by_edge, start, theta_0)
                self.prim.update_prim_count(prim_count)
                swath_cost = self.cost_map[full_swath].sum()
                length = sum(node_path_length)
                # convert nodes in the node path to world coords
                w_node_path = []
                for node in node_path:
                    # convert theta
                    theta = heading_to_world_frame(node[2], full_path[2][0], self.prim.num_headings)
                    w_node_path.append([node[0], node[1], theta])
                node_path = w_node_path

                # initialize variable for nodes added from smoothing
                new_nodes = []
                node_path_smth = []

                # transform to global frame
                full_path[1] += lower_lim
                temp = np.zeros_like(self.cmap.cost_map, dtype=bool)
                temp[lower_lim: upper_lim] = full_swath
                full_swath = temp  # better solution would be to return costmap subset along with swath
                node_path = np.asarray(node_path).T
                node_path[1] += lower_lim
                
                # final check
                for node in full_path.T:
                    if node[0] < 1 or node[0] > (self.cost_map.shape[1] - 1):
                        return False

                print('path length {}'.format(length))

                self.planing_instance += 1

                # return full path and swath
                # original node path
                # smoothed path and added nodes (these are None if smoothing is disabled)
                # list of expanded nodes, g score, swath cost, and path length
                return (full_path, full_swath), \
                       (node_path, node_path_length), \
                       (node_path_smth, new_nodes), \
                       (None, g_score[goal], swath_cost, length, edge_seq, None, None, None, None)

            # find the base heading
            base_heading = node[2] % self.prim.num_base_h
            origin = (0, 0, base_heading)

            # get the edge set based on the current node heading
            edge_set = self.prim.edge_set_dict[origin]

            neighbors = []
            for e in edge_set:
                if e not in self.rotated_prims:
                    self.rotated_prims[e] = (R[0][0] * e[0] + R[0][1] * e[1], R[1][0] * e[0] + R[1][1] * e[1], e[2])
                neighbour = self.concat(node, self.rotated_prims[e], base_heading, self.prim.spacing)
                
                neighbour_heading = heading_to_world_frame(neighbour[2], theta_0, self.prim.num_headings)
                if neighbour_heading > 0 and neighbour_heading < np.pi and neighbour[1] >= node[1] and \
                    0 < neighbour[0] < self.cost_map.shape[1] and 0 < neighbour[1] < self.cmap.shape[0]:
                        
                    # get swath and swath cost
                    key = (e, int(node[2]))
                    if key not in self.swath_dict:
                        self.swath_dict[key] = rotate_swath(self.orig_swath_dict[key], theta_0)
                        self.swath_arg_dict[key] = np.argwhere(self.swath_dict[key] == 1)
                        self.swath_dict_no_pad[key] = rotate_swath(self.orig_swath_dict_no_pad[key], theta_0)
                        self.swath_arg_dict_no_pad[key] = np.argwhere(self.swath_dict_no_pad[key] == 1)

                    temp_path_length = self.prim.path_lengths[(origin, e)]
                    neighbors.append((neighbour, e, temp_path_length))
                    
            if len(neighbors) == 0:
                node = came_from[node]
                continue
            assert len(neighbors) != 0, print("Invalid! No neighbors for node: ", node, "; theta 0: ", theta_0)
            
            neighbour, e, temp_path_length = neighbors[np.random.randint(0, len(neighbors))]
            came_from[neighbour] = copy.deepcopy(node)
            came_from_by_edge[neighbour] = (origin, e)
            path_length[neighbour] = temp_path_length

            node = neighbour
            iter += 1

        print('Failed to find a path!')
        self.diagnostics = {'start': start,
                            'goal': goal_y,
                            'limits': (lower_lim, upper_lim),
                            'cost_map': self.cost_map}
        return False

    def build_path(self, path, came_from_by_edge, start, theta_0) -> Tuple[np.ndarray, np.ndarray, dict, list]:
        """
        Path returned from graph search only consists of nodes between edges
        Need to construct the path via the primitive paths from these nodes
        """
        full_path = []
        full_swath = np.zeros_like(self.cost_map, dtype=bool)
        pt_a = start
        edge_seq = []
        prim_count = {k: {} for k in self.prim.edge_set_dict}
        for pt_b in path[1:]:
            key = came_from_by_edge[pt_b]
            edge_seq.append(key)
            path_ab = self.prim.paths[key]
            origin, edge = key
            theta = heading_to_world_frame(pt_a[2] - origin[2], theta_0, self.prim.num_headings)

            # rotate
            rot_path_ab = self.prim.rotate_path(path_ab, theta)

            # add start point
            rot_path_ab[0] += pt_a[0]
            rot_path_ab[1] += pt_a[1]
            full_path.append(rot_path_ab)

            # add swath
            swath = self.get_swath(pt_a, self.swath_dict[(edge, int(pt_a[2]))], self.cost_map, self.max_val)

            # aggregating swaths themselves is fine since
            # the swath array is of type bool
            full_swath += swath

            # for debugging purposes, keep track of the prims picked
            prim_count[origin][edge] = prim_count[origin][edge] + 1 if key in prim_count[origin] else 1

            # update start point
            pt_a = pt_b

        return np.hstack(full_path), full_swath, prim_count, edge_seq

    @staticmethod
    @jit(nopython=True, cache=True, fastmath=True)
    def get_swath(start_pos, raw_swath, cost_map, max_val) -> np.ndarray:
        # swath mask has starting node at the centre and want to put at the starting node of currently expanded node
        # in the cmap, need to remove the extra columns/rows of the swath mask
        swath_size = raw_swath.shape[0]
        min_y = int(start_pos[1]) - max_val
        max_y = int(start_pos[1]) + max_val + 1
        min_x = int(start_pos[0]) - max_val
        max_x = int(start_pos[0]) + max_val + 1

        # Too close to the bottom
        a0 = 0
        if min_y < 0:
            a0 = abs(min_y)
            min_y = 0

        # Too close to the top
        b0 = swath_size
        if max_y > cost_map.shape[0]:
            b0 = swath_size - (max_y - (cost_map.shape[0]))
            max_y = cost_map.shape[0]

        # Too far to the left
        a1 = 0
        if min_x < 0:
            a1 = abs(min_x)
            min_x = 0

        # Too far to the right
        b1 = swath_size
        if max_x > cost_map.shape[1]:
            b1 = swath_size - (max_x - (cost_map.shape[1]))
            max_x = cost_map.shape[1]

        # fit raw swath onto costmap centred at start_pos
        swath = np.zeros_like(cost_map, dtype=numba.boolean)
        swath[min_y:max_y, min_x:max_x] = raw_swath[a0:b0, a1:b1]

        # compute cost
        return swath


    @staticmethod
    @jit(nopython=True, cache=True, fastmath=True)
    def concat(x: Tuple, y: Tuple, base_heading: int, spacing: float) -> Tuple:
        """
        given two points x,y in the lattice, find the concatenation x + y
        """
        # find the position and heading of the two points
        p1_theta = x[2] * spacing - spacing * base_heading  # starting heading
        p2_theta = y[2] * spacing  # edge heading

        result = [x[0] + (np.cos(p1_theta) * y[0] - np.sin(p1_theta) * y[1]),
                  x[1] + (np.sin(p1_theta) * y[0] + np.cos(p1_theta) * y[1])]

        # compute the final heading after concatenating x and y
        heading = (p2_theta + x[2] * spacing - spacing * base_heading) % M__2_PI

        return result[0], result[1], int(heading / spacing)
