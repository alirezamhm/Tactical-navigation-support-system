import argparse
import os, sys
import time

import numpy as np
from matplotlib import pyplot as plt
import torch

from networks.model import IceModel
from submodules.src.path_evaluator import PredictivePathEvaluator
from submodules.src.primitives import Primitives
from submodules.src.ship import Ship
from submodules.src.swath import generate_swath, view_all_swaths
from submodules.src.utils.plot import Plot
from submodules.src.utils.utils import Path
from submodules.src.occupancy_grid.occupancy_map import OccupancyGrid
from submodules.src.cost_map_occ import CostMap_Occupancy
from submodules.src.a_star_predictive import AStarPredictive
from submodules.src.utils.utils import DotDict
import threading
import copy

# ROS Humble Related
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, MultiArrayDimension, Int32
from geometry_msgs.msg import Pose2D, Point, Polygon, Point32
from custom_msgs.msg import PolygonArray



class PredictiveNode(Node):

    def __init__(self, cfg, start_trial_idx=0, id='1', conc=20, debug=False):
        super().__init__('predictive_node')

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
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("Using device: ", self.device)
        
        self.conc = conc

        self.debug = debug


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
        self.occ_map = self.occ_map[:, :, 0] # only uses con channel

    
    def predictive_planner(self, cfg):

        if cfg.planner != 'predictive':
            raise Exception("Wrong planner. Please run the planner node specified in the config file.")
        
        costmap = CostMap_Occupancy(cfg=cfg, horizon=cfg.a_star.horizon,
                        ship_mass=cfg.ship.mass, **cfg.costmap)

        ship = Ship(scale=cfg.costmap.scale, **cfg.ship)
        prim = Primitives(cache=False, **cfg.prim)
        swath_dict = generate_swath(ship, prim, cache=False,  model_inference=False)
        debug_ice_model = False

        print("Running Predictive Planner")
        ship_no_padding = Ship(scale=cfg.costmap.scale, vertices=cfg.ship.vertices, padding=0, mass=cfg.ship.mass)
        swath_dict_no_padding = generate_swath(ship_no_padding, prim, cache=False, model_inference=True)

        occupancy = OccupancyGrid(grid_width=cfg.occ.grid_size, grid_height=cfg.occ.grid_size, map_width=cfg.occ.map_width, map_height=cfg.occ.map_height, ship_body=None)

        checkpoint = torch.load(cfg.model_path, map_location=self.device)
        hparams = checkpoint['hyper_parameters'] 
        args = argparse.Namespace(**hparams)
        args.arch = "unet_occ"
        args.reg_loss_fn = args.cls_loss_fn = None
        ice_model = IceModel.load_from_checkpoint(cfg.model_path, map_location=self.device, args=args)
        ice_model.eval()

        a_star = AStarPredictive(cmap=costmap,
                ke_map=None,
                prim=prim,
                ship=ship,
                swath_dict=swath_dict,
                swath_dict_no_padding=swath_dict_no_padding,
                ship_no_padding=ship_no_padding,
                ice_model=ice_model,
                diff_scale=cfg.diff_scale,
                device=self.device,
                window_size=(cfg.model_win_h, cfg.model_win_w),
                **cfg.a_star)
    
        path_eval = PredictivePathEvaluator(prim=prim, cmap_scale=cfg.costmap.scale, ice_model=ice_model, diff_scale=cfg.diff_scale,
                                            window_size=(cfg.model_win_h, cfg.model_win_w), device=self.device)

        path_obj = Path()

        replan_count = 0
        compute_time = []
        num_model_calls = []
        last_goal_y = np.inf

        initial_plan_success = False

        prediction_horizon = None
        
        costs_and_lengths = []

        # start main planner loop
        print("planner ROS Running...")
        while replan_count < cfg.get('max_replan', np.infty) and rclpy.ok():

            while rclpy.ok() and ((self.ship_pos is None) or (self.goal is None) or (self.occ_map is None and self.obs is None) or (self.sim_trial_idx is None)):
                if self.sim_trial_idx == -1: # simulation ended
                    break
                self.wait_rate.sleep()

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
            
            elif (not cfg.a_star.replan) and initial_plan_success:
                print("Initial plan done without replan! Waiting...")
                self.wait_rate.sleep()

            # start timer
            t0 = time.time()

            # get ice model observation
            footprint = None

            occupancy.compute_ship_footprint_planner(ship_state=self.ship_pos, ship_vertices=cfg.ship.vertices)
            footprint = np.copy(occupancy.footprint)

            # stop planning if the remaining total distance is less than a ship length in meter
            if self.goal[1] - self.ship_pos[1] <= 2:
                continue

            if self.goal is not None:
                goal_y = min(self.goal[1], (self.ship_pos[1] + cfg.a_star.horizon)) * cfg.costmap.scale
                last_goal_y = self.goal[1]
            else:
                goal_y = min(last_goal_y, (self.ship_pos[1] + cfg.a_star.horizon)) * cfg.costmap.scale

            costmap.update(occ_map=self.occ_map)
            
            if self.prev_replan_pos is not None:
                dist_passed = ((self.ship_pos[0] - self.prev_replan_pos[0])**2 + (self.ship_pos[1] - self.prev_replan_pos[1])**2)**(0.5)
                if dist_passed < self.replan_dist:
                    continue

            # compute path to goal
            ship_pos = copy.deepcopy(self.ship_pos_scaled)
            self.prev_replan_pos = copy.deepcopy(self.ship_pos)
            print("ship position: ", self.prev_replan_pos[:2], "; start planning...")
            occ_map = np.copy(self.occ_map)
            plan_start = time.time()
            search_result = a_star.search(
                start=(ship_pos[0], ship_pos[1], ship_pos[2]),
                goal_y=goal_y,
                occ_map=occ_map,
                centroids=None,
                footprint=footprint,
                ship_vertices=cfg.ship.vertices, 
                debug=debug_ice_model, 
                prediction_horizon=prediction_horizon, 
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
                print("Planning success: ", replan_count)
                initial_plan_success = True
            
            (full_path, full_swath), \
            (node_path, node_path_length), \
            (node_path_smth, new_nodes), \
            (nodes_expanded, g_score, swath_cost, length, edge_seq, swath_costs, swath_ins, horizontal_shifts, path_len_keys) = search_result
  
            costs_and_lengths.append((swath_cost, length))
            print(f"Swath costs sum: {swath_cost}, length: {length}, cost/length: {swath_cost/length}")
            print(f"Swath costs {swath_costs}")

            if path_obj.node_path is not None:

                old_swath_costs = path_eval.eval_path(occ_map=np.copy(self.occ_map), node_path=path_obj.node_path.T, ship_pose=self.ship_pos_scaled, 
                                                    swath_ins=path_obj.swath_ins, horizontal_shifts=path_obj.horizontal_shifts, ship_vertices=cfg.ship.vertices,
                                                    path_len_keys=path_obj.path_len_keys, debug=False)
                
                new_swath_costs = path_eval.eval_path(occ_map=np.copy(self.occ_map), node_path=node_path.T, ship_pose=self.ship_pos_scaled, 
                                                    swath_ins=swath_ins, horizontal_shifts=horizontal_shifts, ship_vertices=cfg.ship.vertices,
                                                    path_len_keys=path_len_keys, debug=False)

            else:
                old_swath_costs = None
                new_swath_costs = None
            
            send_new_path, old_cost, new_cost = path_obj.update_occDiff(old_swath_costs=old_swath_costs, node_path=node_path.T, swath_costs=new_swath_costs, ship_pos=self.ship_pos_scaled,
                                            threshold_dist=cfg.get('threshold_dist', 0) * length,
                                            threshold_cost=cfg.get('threshold_cost'), 
                                            drift_threshold=5)
            print("send path: ", send_new_path, "; old cost: ", old_cost, "; new cost: ", new_cost)

            if send_new_path:
                path_obj.node_path = node_path
                path_obj.swath_ins = swath_ins
                path_obj.horizontal_shifts = horizontal_shifts
                path_obj.path_len_keys = path_len_keys
                path_obj.path = full_path

            path_true_scale = np.c_[(path_obj.path[:2] / cfg.costmap.scale).T, path_obj.path[2]] 

            if send_new_path:
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
        if self.debug:
            np.save(f"outputs/predictive/costs_and_lengths_{self.conc}.npy", np.array(costs_and_lengths))
            print(f"mean cost {np.mean([c for c, l in costs_and_lengths])}, mean cost_to_length_ratio {np.mean([c/l for c, l in costs_and_lengths])}")


if __name__ == '__main__':
    rclpy.init(args=None)

    parser = argparse.ArgumentParser(description="Run the planner.")
    parser.add_argument('--cfg', type=str, required=True, help="Path to the configuration file.")
    parser.add_argument('--trial_range', type=int, nargs=2, required=True, help="Trial range as two integers, e.g., 0 50.")
    parser.add_argument('--conc', type=int, default=20, choices=[20, 30, 40, 50], required=True, help="Concentration of ice in percentage (20, 30, 40, or 50).")
    parser.add_argument('--id', type=str, default='1', help="Instance id of the run to create a unique communication channel.")
    parser.add_argument('--debug', action='store_true', help="Enable debug mode.")

    args = parser.parse_args()
    
    cfg_file = args.cfg
    cfg = cfg = DotDict.load_from_file(cfg_file)

    node = PredictiveNode(cfg=cfg, start_trial_idx=args.trial_range[0], id=args.id, conc=args.conc, debug=args.debug)

    thread = threading.Thread(target=rclpy.spin, args=(node, ), daemon=True)
    thread.start()

    try:
        node.predictive_planner(cfg=cfg)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
