""" A path evaluator by using the trained prediction models """
import os, sys

import numpy as np

from networks.model import IceModel
from submodules.src.primitives import Primitives
from submodules.src.ship import Ship
from submodules.src.swath import Swath
from submodules.src.occupancy_grid.ice_model_utils import crop_window, stitch_window, encode_swath, compute_ship_footprint_planner, view_swath, update_costmap, boundary_cost, get_boundary_map
import torch
from matplotlib import pyplot as plt
from torch import nn

class PredictivePathEvaluator:
    def __init__(self, prim: Primitives, cmap_scale, ice_model: IceModel, diff_scale: float, window_size: tuple[int, int], device=None,):
        self.prim = prim
        self.cmap_scale = cmap_scale

        self.diff_criterion = nn.MSELoss()  # allow reduction here as we are not doing batch prediction

        self.vertical_shift = 0 
        self.win_h, self.win_w = window_size
        self.ice_model = ice_model
        self.diff_scale = diff_scale
        self.device = device if device is not None else torch.device("cpu")
            

    def eval_path(self, occ_map, node_path, ship_pose, swath_ins, horizontal_shifts, ship_vertices, path_len_keys, debug=False):
        """
        This function computes a predictive evaluation of a path. Given a path, current ship position, and 
        the current occ observation, returns the path cost based on predictive occ diff
        :param occ_map: given occupancy observation, assume to be most recent
        :param node_path: assume to be of shape (n_nodes, 3), 3 --> (x, y, theta), where (x, y) is in global costmap frame
        :param ship_pose: current ship position in (x, y, theta) where (x, y) is in global costmap frame
        """

        swath_costs = []
        for i in range(node_path.shape[0] - 1):
            node_src = node_path[i]
            node_target = node_path[i + 1]

            # if ship already possed this segment (e.g. above both start & end), cost 0
            if ship_pose[1] > node_src[1] and ship_pose[1] > node_target[1]:
                cost = 0
            
            # all other cases, compute cost
            else:
                swath_in = swath_ins[i]
                horizontal_shift = horizontal_shifts[i]

                # compute global footprint
                transformed_node = (node_src[0], node_src[1], 0)
                theta_0 = np.pi / 2
                global_footprint = compute_ship_footprint_planner(node=transformed_node, theta_0=theta_0, ship_vertices=ship_vertices, occ_map_height=occ_map.shape[0], occ_map_width=occ_map.shape[1], scale=self.cmap_scale)

                occ_map_in, x_low_map, x_high_map, y_low_map, y_high_map, x_low_win, x_high_win, y_low_win, y_high_win = crop_window(occ_map, node_src, win_width=self.win_w, win_height=self.win_h, horizontal_shift=horizontal_shift, vertical_shift=self.vertical_shift)
                footprint_in, _, _, _, _, _, _, _, _ = crop_window(global_footprint, node_src, win_width=self.win_w, win_height=self.win_h, horizontal_shift=horizontal_shift, vertical_shift=self.vertical_shift)

                x = np.concatenate((np.array([occ_map_in]), np.array([footprint_in]), np.array([swath_in])))
                x = torch.Tensor(x)     # (3 x W x H)
                x = x.unsqueeze(dim=0)      # (1 x 3 x W x H)
                x = x.to(self.device)

                # obtain prediction
                y = self.ice_model(x)
                occ_map_hat = y                                         # (1 x 1 x W x H)
                occ_map_hat = occ_map_hat.squeeze().detach()        # (W x H)

                # compute occ diff
                occ_before = x[0, 0, :, :]      # (W, H)
                occ_diff = self.diff_criterion(occ_map_hat, occ_before)
                occ_map_hat = occ_map_hat.cpu().numpy()

                # compute swath cost
                swath_cost = occ_diff.item() * self.diff_scale
                origin, e = path_len_keys[i]
                temp_path_length = self.prim.path_lengths[(origin, e)]
                cost = swath_cost + temp_path_length

                # update global occ map
                occ_map = stitch_window(occ_map, occ_map_hat, x_low_map, x_high_map, y_low_map, y_high_map, x_low_win, x_high_win, y_low_win, y_high_win)


                if debug:
                    self.save_debug_plots(i, occ_map, global_footprint, occ_map_in, occ_map_hat, swath_in)
            
            swath_costs.append(cost)

        return swath_costs
    

    def save_debug_plots(self, i, occ_map, global_footprint, occ_map_in, occ_map_hat, swath_in):
        fig, ax = plt.subplots()
        save_dir = "outputs/predictive/path_debug/"

        ax.clear()
        occ_map_render = np.flip(occ_map, axis=0)
        ax.imshow(occ_map_render, cmap='gray')
        fig.savefig(save_dir + str(i) + "_occ_map.png", bbox_inches='tight', transparent=False, pad_inches=0)

        ax.clear()
        footprint_render = np.flip(global_footprint, axis=0)
        ax.imshow(footprint_render, cmap='gray')
        fig.savefig(save_dir + str(i) + "_footprint.png", bbox_inches='tight', transparent=False, pad_inches=0)

        ax.clear()
        occ_map_in_render = np.flip(occ_map_in, axis=0)
        ax.imshow(occ_map_in_render, cmap='gray')
        fig.savefig(save_dir + str(i) + "_occ_map_in.png", bbox_inches='tight', transparent=False, pad_inches=0)
        
        ax.clear()
        occ_map_hat_render = np.flip(occ_map_hat, axis=0)
        ax.imshow(occ_map_hat_render, cmap='gray')
        fig.savefig(save_dir + str(i) + "_occ_map_hat.png", bbox_inches='tight', transparent=False, pad_inches=0)

        ax.clear()
        swath_in_render = np.flip(swath_in, axis=0)
        ax.imshow(swath_in_render, cmap='gray')
        fig.savefig(save_dir + str(i) + "_swath_in.png", bbox_inches='tight', transparent=False, pad_inches=0)
        


class IceNetPathEvaluator:
    def __init__(self, prim: Primitives, cmap_scale, ice_model: IceModel, cost_scale: float, cost_log: bool, max_thickness: float, max_velocity: float, window_size: tuple[int, int], device=None,):
        self.prim = prim
        self.cmap_scale = cmap_scale

        self.vertical_shift = 0 
        self.win_h, self.win_w = window_size
        self.ice_model = ice_model
        self.cost_scale = cost_scale
        self.cost_log = cost_log
        self.max_thickness = max_thickness
        self.max_velocity = max_velocity
        self.device = device if device is not None else torch.device("cpu")

    def eval_path(self, occ_map, node_path, ship_pose, swath_ins, horizontal_shifts, ship_vertices, path_len_keys, debug=False):
        """
        This function computes a predictive evaluation of a path. Given a path, current ship position, and 
        the current occ observation, returns the path cost based on cost prediction
        :param occ_map: given occupancy observation, assume to be most recent
        :param node_path: assume to be of shape (n_nodes, 3), 3 --> (x, y, theta), where (x, y) is in global costmap frame
        :param ship_pose: current ship position in (x, y, theta) where (x, y) is in global costmap frame
        """

        swath_costs = []
        for i in range(node_path.shape[0] - 1):
            node_src = node_path[i]
            node_target = node_path[i + 1]

            # if ship already possed this segment (e.g. above both start & end), cost 0
            if ship_pose[1] > node_src[1] and ship_pose[1] > node_target[1]:
                cost = 0
            
            # all other cases, compute cost
            else:
                swath_in = swath_ins[i]
                horizontal_shift = horizontal_shifts[i]

                # compute global footprint
                transformed_node = (node_src[0], node_src[1], 0)
                theta_0 = np.pi / 2
                global_footprint = compute_ship_footprint_planner(node=transformed_node, theta_0=theta_0, ship_vertices=ship_vertices, occ_map_height=occ_map.shape[0], occ_map_width=occ_map.shape[1], scale=self.cmap_scale)

                occ_map_in, x_low_map, x_high_map, y_low_map, y_high_map, x_low_win, x_high_win, y_low_win, y_high_win = crop_window(occ_map, node_src, win_width=self.win_w, win_height=self.win_h, horizontal_shift=horizontal_shift, vertical_shift=self.vertical_shift)
                footprint_in, _, _, _, _, _, _, _, _ = crop_window(global_footprint, node_src, win_width=self.win_w, win_height=self.win_h, horizontal_shift=horizontal_shift, vertical_shift=self.vertical_shift)

                occ_map_in = np.transpose(occ_map_in, (2, 0, 1)) # (W, H, 4) -> (4, W, H)
                x = np.concatenate((occ_map_in, np.array([footprint_in]), np.array([swath_in])))
                x = torch.Tensor(x)     # (6 x W x H)
                x = x.unsqueeze(dim=0)      # (1 x 6 x W x H)
                x[:, 1, : , :] = x[:, 1, :, :] / self.max_thickness
                x[:, 2, : , :] = x[:, 2, :, :] / self.max_velocity
                x[:, 3, : , :] = x[:, 3, :, :] / self.max_velocity
                x = x.to(self.device)

                # obtain prediction
                map_hat, cls_hat, reg_hat = self.ice_model(x)
                map_hat = map_hat.squeeze(dim=0).detach().cpu().numpy()        # (4 x W x H)
                map_hat[1, :, :] = map_hat[1, :, :] * self.max_thickness
                map_hat[2, :, :] = map_hat[2, :, :] * self.max_velocity
                map_hat[3, :, :] = map_hat[3, :, :] * self.max_velocity
                cls_hat = torch.sigmoid(cls_hat).squeeze(dim=0).detach().cpu().numpy()[0]        # (1)
                reg_hat = reg_hat.squeeze(dim=0).detach().cpu().numpy()[0]      # (1)
                
                if cls_hat < 0.5: 
                    cost_hat = 0
                else:
                    if self.cost_log:
                        cost_hat = np.exp(reg_hat)
                    else:
                        cost_hat = np.clip(reg_hat, 0, None)

                # compute swath cost
                swath_cost = cost_hat * self.cost_scale
                
                origin, e = path_len_keys[i]
                temp_path_length = self.prim.path_lengths[(origin, e)]
                cost = swath_cost + temp_path_length

                # update global occ map
                map_hat = np.transpose(map_hat, (1, 2, 0)) # (4, W, H) -> (W, H, 4)
                occ_map = stitch_window(occ_map, map_hat, x_low_map, x_high_map, y_low_map, y_high_map, x_low_win, x_high_win, y_low_win, y_high_win)

                if debug:
                    print("Swath cost: ", swath_cost, " Path length: ", temp_path_length, " Total cost: ", cost)
                    self.save_debug_plots(i, occ_map, global_footprint, occ_map_in, map_hat, swath_in)
            
            swath_costs.append(cost)

        return swath_costs
    
    def save_debug_plots(self, i, occ_map, global_footprint, occ_map_in, occ_map_hat, swath_in):
        save_dir = "outputs/icenet/path_debug"
        os.makedirs(save_dir, exist_ok=True)

        def plot_channels(img, name):
            img = np.flip(img, axis=0)
            if img.ndim == 2:
                plt.figure()
                plt.imshow(img, cmap='gray')
                plt.savefig(f"{save_dir}/{i}_{name}.png", bbox_inches='tight', transparent=False, pad_inches=0)
                plt.close()
            elif img.ndim == 3:
                for ch in range(img.shape[2]):
                    plt.figure()
                    plt.imshow(img[..., ch], cmap='gray')
                    plt.title(f"{name} channel {ch}")
                    plt.savefig(f"{save_dir}/{i}_{name}_ch{ch}.png", bbox_inches='tight', transparent=False, pad_inches=0)
                    plt.close()

        plot_channels(occ_map, "occ_map")
        plot_channels(global_footprint, "footprint")
        plot_channels(np.transpose(occ_map_in, (1, 2, 0)), "occ_map_in")
        plot_channels(occ_map_hat, "occ_map_hat")
        plot_channels(swath_in, "swath_in")