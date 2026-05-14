import os, sys
import time
import argparse

import numpy as np
from matplotlib import pyplot as plt

from submodules.src.a_star_random import AStarRandom
from submodules.src.cost_map import CostMap
from submodules.src.primitives import Primitives
from submodules.src.ship import Ship
from submodules.src.swath import generate_swath, view_all_swaths
from submodules.src.utils.plot import Plot
from submodules.src.utils.utils import Path
from submodules.src.utils.utils import DotDict
import threading
import copy

# ROS Humble Related
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, MultiArrayDimension, Int32
from geometry_msgs.msg import Pose2D, Point, Polygon, Point32
from custom_msgs.msg import PolygonArray


class RandomNode(Node):

    def __init__(self, cfg, start_trial_idx=0, id='1', conc=20):
        super().__init__('random_node')

        self.cfg = cfg

        # publishers
        self.path_publisher = self.create_publisher(Float32MultiArray, f'path_{id}', 1)
        self.cur_trial_idx_pub = self.create_publisher(Int32, f'high_level_trial_idx_{id}', 1)

        # subscribers
        self.occ_sub = self.create_subscription(Float32MultiArray, f'occ_map_{id}', self.occ_callback, 1)
        self.callback_count = 0
        self.occ_map = None

        self.poly_sub = self.create_subscription(PolygonArray, f'polygons_{id}', self.poly_callback, 1)
        self.obs = None

        self.ship_pose_sub = self.create_subscription(Pose2D, f'ship_pose_{id}', self.ship_pose_callback, 1)
        self.ship_pos = None        # ship pose (x, y, theta) in meter scale
        self.ship_pos_scaled = None     # ship pose (x, y, theta) in costmap scale

        self.goal_sub = self.create_subscription(Point, f'goal_{id}', self.goal_callback, 1)
        self.goal = None

        self.trial_idx_sub = self.create_subscription(Int32, f'trial_idx_{id}', self.trial_idx_callback, 1)
        self.cur_trial_idx = start_trial_idx       # current trial the planner is running on
        self.sim_trial_idx = None       # trial the sim node is running on

        self.wait_rate = self.create_rate(5)

        self.replan_dist = cfg.a_star.replan_dist
        self.prev_replan_pos = None

        self.conc = conc


    def poly_callback(self, msg):
        polygons = msg.polygons
        obs = []
        for poly in polygons:
            verts = []
            points = poly.points
            for pt in points:
                verts.append([pt.x, pt.y])
            verts = np.array(verts) 
            
            obs.append(verts)
        self.obs = obs

    def trial_idx_callback(self, msg):
        self.sim_trial_idx = msg.data

    def goal_callback(self, goal_msg):
        self.goal = (goal_msg.x, goal_msg.y)

    def ship_pose_callback(self, pose_msg):
        self.ship_pos = np.array([pose_msg.x, pose_msg.y, pose_msg.theta])
        self.ship_pos_scaled = np.array([pose_msg.x * self.cfg.costmap.scale, pose_msg.y * self.cfg.costmap.scale, pose_msg.theta])
    
    def occ_callback(self, msg):
        height = msg.layout.dim[0].size
        width = msg.layout.dim[1].size

        self.occ_map = np.array(msg.data, dtype=np.float32).reshape((height, width, 4))
        self.occ_map = self.occ_map[:, :, 0] 
    
    def random_planner(self, cfg):

        if cfg.planner != 'random':
            raise Exception("Wrong planner. Please run the planner node specified in the config file.")

        costmap = CostMap(horizon=cfg.a_star.horizon,
                        ship_mass=cfg.ship.mass, **cfg.costmap)
        use_ice_model = False

        ship = Ship(scale=cfg.costmap.scale, **cfg.ship)
        prim = Primitives(cache=False, **cfg.prim)
        swath_dict = generate_swath(ship, prim, cache=False,  model_inference=False)
        debug_ice_model = False

        print("Running Random Planner")
        ship_no_padding = Ship(scale=cfg.costmap.scale, vertices=cfg.ship.vertices, padding=0, mass=cfg.ship.mass)
        swath_dict_no_padding = generate_swath(ship_no_padding, prim, cache=False, model_inference=True)

        a_star = AStarRandom(cmap=costmap,
            prim=prim,
            ship=ship,
            swath_dict=swath_dict,
            swath_dict_no_padding=swath_dict_no_padding,
            ship_no_padding=ship_no_padding,
            use_ice_model=use_ice_model,
            **cfg.a_star)

        path_obj = Path()

        replan_count = 0
        compute_time = []
        last_goal_y = np.inf

        initial_plan_success = False

        print("planner ROS Running...")
        while replan_count < cfg.get('max_replan', np.infty) and rclpy.ok():

            while rclpy.ok() and ((self.ship_pos is None) or (self.goal is None) or (self.occ_map is None and self.obs is None) or (self.sim_trial_idx is None)):
                if self.sim_trial_idx == -1:
                    break
                self.wait_rate.sleep()

            # check if sim node has already start a new trial
            if self.cur_trial_idx != self.sim_trial_idx:
                if self.sim_trial_idx == -1:
                    break
                print("Starting new trial: ", self.sim_trial_idx)
                self.cur_trial_idx = self.sim_trial_idx
                path_obj = Path()
                replan_count = 0
                self.ship_pos = None
                self.goal = None
                self.occ_map = None
                self.obs = None
                initial_plan_success = False
                self.prev_replan_pos = None
                continue
            
            elif rclpy.ok() and (not cfg.a_star.replan) and initial_plan_success:
                self.wait_rate.sleep()

            # start timer
            t0 = time.time()

            footprint = None

            # stop planning if the remaining total distance is less than a ship length in meter
            if self.goal[1] - self.ship_pos[1] <= 2:
                continue

            if self.goal is not None:
                goal_y = min(self.goal[1], (self.ship_pos[1] + cfg.a_star.horizon)) * cfg.costmap.scale
                last_goal_y = self.goal[1]
            else:
                goal_y = min(last_goal_y, (self.ship_pos[1] + cfg.a_star.horizon)) * cfg.costmap.scale


            costmap.update(self.obs, self.ship_pos_scaled[1] - ship.max_ship_length / 2,
                        vs=(cfg.controller.target_speed * cfg.costmap.scale + 1e-8))

            if initial_plan_success and self.prev_replan_pos is not None:
                dist_passed = ((self.ship_pos[0] - self.prev_replan_pos[0])**2 + (self.ship_pos[1] - self.prev_replan_pos[1])**2)**(0.5)
                if dist_passed < self.replan_dist:
                    continue

            ship_pos = copy.deepcopy(self.ship_pos_scaled)   
            self.prev_replan_pos = copy.deepcopy(self.ship_pos)
            print("trial: ", self.sim_trial_idx)
            print("ship position: ", self.prev_replan_pos[:2], "; start planning...")
            occ_map = None
            plan_start = time.time()
            search_result = a_star.search(
                start=(ship_pos[0], ship_pos[1], ship_pos[2]),
                goal_y=goal_y,
                occ_map=occ_map,
                centroids=None,
                footprint=footprint,
                ship_vertices=cfg.ship.vertices, 
                use_ice_model=use_ice_model,
                debug=debug_ice_model, 
            )
            print("planning time: ", time.time() - plan_start)

            if self.cur_trial_idx != self.sim_trial_idx:
                continue
            
            if not search_result:
                print("Planner failed to find a path!")
                replan_count += 1
                self.ship_pos = None
                self.occ_map = None
                continue
            else:
                print("Planning success: ", replan_count+1)
                initial_plan_success = True
            
            (full_path, full_swath), \
            (node_path, node_path_length), \
            (node_path_smth, new_nodes), \
            (nodes_expanded, g_score, swath_cost, length, edge_seq, swath_costs, swath_ins, horizontal_shifts, path_len_keys) = search_result
            
            send_new_path = path_obj.update(full_path, full_swath, costmap.cost_map, self.ship_pos_scaled[1],
                                            threshold_dist=cfg.get('threshold_dist', 0) * length,
                                            threshold_cost=cfg.get('threshold_cost'))

            path_true_scale = np.c_[(path_obj.path[:2] / cfg.costmap.scale).T, path_obj.path[2]] 

            if send_new_path:
                print("sending new path")
                path_msg = Float32MultiArray()
                dim_h = MultiArrayDimension()
                dim_h.label = 'height'
                dim_h.size = path_true_scale.shape[0]
                path_msg.layout.dim.append(dim_h)
                dim_w = MultiArrayDimension()
                dim_w.label = 'width'
                dim_w.size = path_true_scale.shape[1]
                path_msg.layout.dim.append(dim_w)
                path_msg.data = path_true_scale.flatten().tolist()
                self.path_publisher.publish(path_msg)
            else:
                ...

            compute_time.append((time.time() - t0))
            replan_count += 1

            self.ship_pos = None
            self.occ_map = None
            self.obs = None
        print("Finishing")


if __name__ == '__main__':
    rclpy.init(args=None)
    
    parser = argparse.ArgumentParser(description="Run the planner.")
    parser.add_argument('--cfg', type=str, required=True, help="Path to the configuration file.")
    parser.add_argument('--trial_range', type=int, nargs=2, required=True, help="Trial range as two integers, e.g., 0 50.")
    parser.add_argument('--conc', type=int, default=20, choices=[20, 30, 40, 50], required=True, help="Concentration of ice in percentage (20, 30, 40, or 50).")
    parser.add_argument('--id', type=str, default='1', help="Instance id of the run to create a unique communication channel.")

    args = parser.parse_args()
    
    cfg_file = args.cfg
    cfg = cfg = DotDict.load_from_file(cfg_file)

    node = RandomNode(cfg=cfg, start_trial_idx=args.trial_range[0],  conc=args.conc, id=args.id)

    thread = threading.Thread(target=rclpy.spin, args=(node, ), daemon=True)
    thread.start()

    try:
        node.random_planner(cfg=cfg)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
