import pickle
from typing import List
import numpy as np
import cv2
import math
import os
import matplotlib.pyplot as plt
from pymunk import Body, Vec2d
from skimage.draw import draw

class OccupancyGrid:

    def __init__(self, grid_width, grid_height, map_width, map_height, pixels_per_grid=4, ship_body=None) -> None:
        """
        grid_width, grid_height, map_width, map_height are in meter units
        ship body info at starting position
        """
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.map_width = map_width
        self.map_height = map_height
        self.pixels_per_grid = pixels_per_grid
        self.occ_map_width = int(self.map_width / self.grid_width)         # number of grids in x-axis
        self.occ_map_height = int(self.map_height / self.grid_height)      # number of grids in y-axis

        print("Occupancy map resolution: ", grid_width, "; occupancy map dimension: ", (self.occ_map_width, self.occ_map_height))

        self.occ_map = np.zeros((self.occ_map_height, self.occ_map_width, 4))
        self.footprint = np.zeros((self.occ_map_height, self.occ_map_width))
        self.target_encoding = np.zeros((self.occ_map_height, self.occ_map_width))
        self.swath = np.zeros((self.occ_map_height, self.occ_map_width))

        self.con_fig, self.con_ax = plt.subplots(figsize=(10, 10))
        self.con_ax.set_xlabel('')
        self.con_ax.set_xticks([])
        self.con_ax.set_ylabel('')
        self.con_ax.set_yticks([])

        self.local_window_width = 16
        self.local_window_height = 6

        if ship_body is None:
            self.cur_grid_x, self.cur_grid_y = 0, 0
        else:
            self.cur_grid_x, self.cur_grid_y = self.get_current_grid(ship_body=ship_body)

        self.con_target_grid_x = None
        self.con_target_grid_y = None
        self.con_grid_entry = None
        self.con_grid_exit = None
        self.t = 0

        self.con_labels = np.zeros((self.occ_map_height, 6))   # 10 step x (prev_obs, label, target_x, target_y, entry, exit)
        self.flow_field = np.zeros((self.occ_map_height, self.occ_map_height, self.occ_map_width, 2))  # flow field labels (t_dim, y_dim, x_dim, 2)

        self.occ_observations = []
        self.swath_observations = []
        self.footprint_observations = []
        self.cur_grids = []                 # keeps track of cur_grid_x, cur_grid_y in each timestep
        self.metrics = []

        # data collection status
        self.con_collection = False


    def compute_occ_img(self, vertices, thicknesses, velocities):
        ice_data_h = self.occ_map_height * self.pixels_per_grid
        ice_data_w = self.occ_map_width * self.pixels_per_grid
        
        pixel_per_meter = ice_data_h / self.map_height

        raw_ice_data = np.zeros((ice_data_h, ice_data_w, 4))

        for obstacle, thickness, velocity in zip(vertices, thicknesses, velocities):
            obstacle = np.asarray(obstacle) * pixel_per_meter

            # get pixel coordinates on costmap that are contained inside obstacle/polygon
            rr, cc = draw.polygon(obstacle[:, 1], obstacle[:, 0], shape=raw_ice_data.shape)

            # skip if 0 area
            if len(rr) == 0 or len(cc) == 0:
                continue

            raw_ice_data[rr, cc] = [1.0, thickness, velocity[0], velocity[1]]
        
        return raw_ice_data
    

    def compute_con_gridmap(self, raw_ice_data, save_fig_dir=None, visualize=False, t=0):
        """
        Compute concentration grid map
        """
        pixel_per_meter_y = raw_ice_data.shape[0] / self.map_height
        pixel_per_meter_x = raw_ice_data.shape[1] / self.map_width

        for i in range(self.occ_map_height):
            y_low = int(i * self.grid_height * pixel_per_meter_y)
            y_high = int((i + 1) * self.grid_height * pixel_per_meter_y)
            if y_high >= raw_ice_data.shape[0]:
                y_high = raw_ice_data.shape[0] - 1

            for j in range(self.occ_map_width):
                x_low = int(j * self.grid_width * pixel_per_meter_x)
                x_high = int((j + 1) * self.grid_width * pixel_per_meter_x)
                if x_high >= raw_ice_data.shape[1]:
                    x_high = raw_ice_data.shape[1] - 1
                cropped_region = raw_ice_data[y_low:y_high, x_low:x_high]

                self.occ_map[i, j] = np.mean(cropped_region, axis=(0, 1))


        if (save_fig_dir is not None) and visualize:
            if not os.path.exists(save_fig_dir):
                os.makedirs(save_fig_dir)
            occ_map_render = np.copy(self.occ_map)
            occ_map_render = np.flip(occ_map_render, axis=0)
            
            channel_names = ['con', 'thick', 'vel_x', 'vel_y']
            for channel in range(4):
                if channel > 1:
                    self.con_ax.imshow(occ_map_render[:, :, channel], cmap='gray', vmin=-.5, vmax=.5)
                else:
                    self.con_ax.imshow(occ_map_render[:, :, channel], cmap='gray')
                self.con_ax.axis('off')
                fp = os.path.join(save_fig_dir, f"{str(t)}_{channel_names[channel]}.png")
                self.con_fig.savefig(fp, bbox_inches='tight', transparent=False, pad_inches=0)


    def compute_swath(self, body, ship_vertices, padding=0.25):
        meter_to_grid_scale_x = self.occ_map_width / self.map_width
        meter_to_grid_scale_y = self.occ_map_height / self.map_height

        # ship vertices in meter
        heading = body.angle
        R = np.asarray([
            [math.cos(heading), -math.sin(heading)], [math.sin(heading), math.cos(heading)]
        ])
        vertices = np.asarray(ship_vertices) @ R.T + np.asarray(body.position)

        r = []
        c = []
        for x, y in vertices:
            grid_x = x * meter_to_grid_scale_x
            grid_y = y * meter_to_grid_scale_y
            if grid_y < 0 or grid_y >= self.occ_map_height or grid_x < 0 or grid_x >= self.occ_map_width:
                continue
            r.append(grid_y)
            c.append(grid_x)

        rr, cc = draw.polygon(r=r, c=c)
        self.swath[rr, cc] = 1.0


    def compute_ship_footprint(self, body, ship_vertices, padding=0.25):
        self.footprint = np.zeros((self.occ_map_height, self.occ_map_width))
        meter_to_grid_scale_x = self.occ_map_width / self.map_width
        meter_to_grid_scale_y = self.occ_map_height / self.map_height

        # ship vertices in meter
        heading = body.angle
        R = np.asarray([
            [math.cos(heading), -math.sin(heading)], [math.sin(heading), math.cos(heading)]
        ])
        vertices = np.asarray(ship_vertices) @ R.T + np.asarray(body.position)

        r = []
        c = []
        for x, y in vertices:
            grid_x = x * meter_to_grid_scale_x
            grid_y = y * meter_to_grid_scale_y
            if grid_y < 0 or grid_y >= self.occ_map_height or grid_x < 0 or grid_x >= self.occ_map_width:
                continue
            r.append(grid_y)
            c.append(grid_x)

        rr, cc = draw.polygon(r=r, c=c)
        self.footprint[rr, cc] = 1.0
        
        return np.array([rr, cc]).T

    
    def compute_ship_footprint_planner(self, ship_state, ship_vertices, padding=0.25):
        """
        NOTE this function computes current ship footprint similarily to self.compute_ship_footprint()
        but is intended for generating observations for planners 
        :param ship_state: (x, y, theta) where x, y are in meter and theta in radian
        :param ship_vertices: original unscaled, unpadded ship vertices
        """
        self.footprint = np.zeros((self.occ_map_height, self.occ_map_width))
        meter_to_grid_scale_x = self.occ_map_width / self.map_width
        meter_to_grid_scale_y = self.occ_map_height / self.map_height

        position = ship_state[:2]
        angle = ship_state[2]

        # ship vertices in meter
        heading = angle
        R = np.asarray([
            [math.cos(heading), -math.sin(heading)], [math.sin(heading), math.cos(heading)]
        ])
        vertices = np.asarray(ship_vertices) @ R.T + np.asarray(position)

        r = []
        c = []
        for x, y in vertices:
            grid_x = x * meter_to_grid_scale_x
            grid_y = y * meter_to_grid_scale_y
            if grid_y < 0 or grid_y >= self.occ_map_height or grid_x < 0 or grid_x >= self.occ_map_width:
                continue
            r.append(grid_y)
            c.append(grid_x)
        
        # it is possible that the ship state is outside of the grid map
        if len(r) == 0 or len(c) == 0:
            print("ship outside the costmap!!!")
            return
        
        rr, cc = draw.polygon(r=r, c=c)
        
        self.footprint[rr, cc] = 1.0


    def encode_target_grid(self, save_fig_dir):
        self.target_encoding = np.zeros((self.occ_map_height, self.occ_map_width))

        if self.cur_grid_y >= self.occ_map_height or self.cur_grid_x >= self.occ_map_width:
            return
        
        self.target_encoding[self.cur_grid_y, self.cur_grid_x] = 1.0

    def save_snapshot(self, obs_vertices: List[np.array], thicknesses: List[float], velocities: List[np.array],
                      ship_body: Body, ship_vertices: List[Vec2d], metrics: List[float]):
        """
        Saves a snapshot of the current state of the occupancy map, swath, ship footprint, 
        and other relevant metrics for trial data extraction.
        Args:
            obs_vertices (List[np.array]): List of obstacle vertices.
            thicknesses (List[float]): List of thickness values for the obstacles.
            velocities (List[np.array]): List of velocity vectors for the obstacles.
            ship_body (Body): The body of the ship.
            ship_vertices (List[Vec2d]): List of vertices defining the ship's shape.
            metrics (List[float]): List of physical metrics.
        """
        
        raw_ice_data = self.compute_occ_img(obs_vertices, thicknesses, velocities)
        self.compute_con_gridmap(raw_ice_data=raw_ice_data)
        self.occ_observations.append(np.copy(self.occ_map))
        
        footprint = self.compute_ship_footprint(ship_body, ship_vertices)
        self.footprint_observations.append(np.copy(footprint))
        
        previous_swath = self.swath_observations[-1] if self.swath_observations else None
        swath = np.concatenate((previous_swath, footprint), axis=0) if previous_swath is not None else footprint
        self.swath_observations.append(np.copy(swath))
        
        cur_grid_x, cur_grid_y = self.get_current_grid(ship_body=ship_body)
        self.cur_grids.append([cur_grid_x, cur_grid_y])
        
        self.metrics.append(metrics)
        
    def export(self, dir: str, data_size: int=None):
        """
        Exports the occupancy map data to the specified directory.
        This method saves various observations and metrics related to the occupancy map in the given directory.
        If the length of the lists is greater than `data_size`, it randomly selects `data_size` indices. If the length is less
        than `data_size`, it repeats the last data to reach the required size, this is used to make sure the
        number of saved data is consistent across all trials.
        Args:
            dir (str): The directory where the trial data will be exported.
            data_size (int): The number of snapshots to be saved. If None, all data will be saved.
        """
        
        os.makedirs(dir, exist_ok=True)
        
        data_length = len(self.occ_observations)
        if not data_size:
            data_size = data_length
            
        if data_length > data_size:
            indices = sorted(np.random.choice(data_length, data_size, replace=False))
        else:
            indices = list(range(data_length)) + [data_length - 1] * (data_size - data_length)

        data_to_save = {
            'occ.npz': [self.occ_observations[i] for i in indices],
            'planner.pkl': {
                'footprint': [self.footprint_observations[i] for i in indices],
                'swath': [self.swath_observations[i] for i in indices],
                'cur_grids': [self.cur_grids[i] for i in indices],
                'metrics': [self.metrics[i] for i in indices],
            },
        }

        # Save each dataset to its corresponding file
        for filename, data in data_to_save.items():
            if 'npy' in filename:
                np.save(os.path.join(dir, filename), np.array(data))
            elif 'npz' in filename:
                np.savez_compressed(os.path.join(dir, filename), data=np.array(data))
            else:
                with open(os.path.join(dir, filename), 'wb') as f:
                    pickle.dump(data, f)

    def clear_data(self):
        """
        Clears the data stored in the occupancy map.
        """
        self.occ_map = np.zeros((self.occ_map_height, self.occ_map_width, 4))
        self.footprint = np.zeros((self.occ_map_height, self.occ_map_width))
        self.swath = np.zeros((self.occ_map_height, self.occ_map_width))
        
        self.occ_observations = []
        self.swath_observations = []
        self.footprint_observations = []
        self.cur_grids = []
        self.metrics = []

    def get_current_grid(self, ship_body):
        return (int(ship_body.position.x // self.grid_width), int(ship_body.position.y // self.grid_height))