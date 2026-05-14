""" Main script for running simulation experiments with autonomous ship navigation in ice """
import os
from typing import List
import pickle
from operator import le, ge
import argparse
import time

import matplotlib
import numpy as np
import pymunk
from pymunk import Vec2d
import threading
from matplotlib import pyplot as plt

from submodules.src.controller.dp import DP
from submodules.src.cost_map import CostMap
from submodules.src.evaluation.metrics import tracking_error, total_work_done
from submodules.src.geometry.polygon import poly_area
from submodules.src.ship import Ship
from submodules.src.utils.plot import Plot
from submodules.src.utils.sim_utils import apply_currents, generate_sim_obs
from submodules.src.utils.utils import DotDict
from submodules.src.occupancy_grid.occupancy_map import OccupancyGrid

# ROS Humble Related
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, MultiArrayDimension, Int32
from geometry_msgs.msg import Pose2D, Point, Polygon, Point32
from custom_msgs.msg import PolygonArray


class Sim2DNode(Node):

    def __init__(self, cfg=None, env_file=None, output_dir=None, conc=None, trial_range=None, id='1'):
        super().__init__('simulation_node')

        self.occupancy = OccupancyGrid(grid_width=cfg.occ.grid_size, grid_height=cfg.occ.grid_size,
                                       map_width=cfg.occ.map_width, map_height=cfg.occ.map_height,
                                       pixels_per_grid=cfg.occ.pixels_per_grid, ship_body=None)
        self.cfg = cfg
        self.env_file = env_file
        self.trial_range = trial_range
        self.output_dir = output_dir
        self.conc = conc
        
        os.makedirs(self.output_dir, exist_ok=True)

        # publishers
        self.occ_publisher = self.create_publisher(Float32MultiArray, f'occ_map_{id}', 1)
        self.ship_pose_publisher = self.create_publisher(Pose2D, f'ship_pose_{id}', 1)
        self.goal_publisher = self.create_publisher(Point, f'goal_{id}', 1)
        self.trial_idx_publisher = self.create_publisher(Int32, f'trial_idx_{id}', 1)
        self.poly_publisher = self.create_publisher(PolygonArray, f'polygons_{id}', 1)

        # subscribers
        self.path = self.create_subscription(Float32MultiArray, f'path_{id}', self.path_callback, 1)
        self.path = None
        self.new_path = None
        self.new_path_received = False
        
        # frequency control
        self.wait_path_rate = self.create_rate(5)
        self.rate_dt = self.create_rate(50)         # based on controller dt = 0.02
        
        if not cfg.anim.show:
            plt.ioff()
            matplotlib.use('Agg')
    

    def path_callback(self, path_msg):
        # get dimensions from layout
        height = path_msg.layout.dim[0].size
        width = path_msg.layout.dim[1].size

        # convert flat data back to occ map
        self.new_path = np.array(path_msg.data, dtype=np.float32).reshape((height, width))
        self.new_path_received = True
        
    def compute_occupancy_map(self, vertices: List[np.array], thicknesses: List[float], velocities: List[np.array], trial_idx=0, t=0):
        raw_ice_data = self.occupancy.compute_occ_img(vertices, thicknesses, velocities)
        self.occupancy.compute_con_gridmap(raw_ice_data=raw_ice_data)

    def publush_planner_msg(self, vertices: List[np.array], thicknesses: List[float], velocities: List[np.array], trial_idx=0, t=0):
        # if running baseline, send polygons
        if thicknesses and velocities:
            assert len(vertices) == len(thicknesses) == len(velocities), \
                "The lengths of obstacles, thicknesses, and velocities must be the same."
                
        if cfg.planner == 'skeleton' or cfg.planner == 'straight' or cfg.planner == 'lattice' or cfg.planner == 'random':
            poly_msg = PolygonArray()
            ros_polygons = []
            for obstacle in vertices:
                poly = Polygon()
                points = []
                for vert in obstacle:
                    pt = Point32()
                    pt.x = vert[0]
                    pt.y = vert[1]
                    points.append(pt)
                poly.points = points
                ros_polygons.append(poly)
            
            poly_msg.polygons = ros_polygons
            self.poly_publisher.publish(poly_msg)
            return poly_msg

        elif cfg.planner == 'predictive' or cfg.planner == 'icenet':
            # prepare obstacle occupancy
            self.compute_occupancy_map(vertices, thicknesses, velocities, trial_idx=trial_idx, t=t)
            occ_map = np.copy(self.occupancy.occ_map)

            msg = Float32MultiArray()
            dim_h = MultiArrayDimension()
            dim_h.label = 'height'
            dim_h.size = occ_map.shape[0]
            msg.layout.dim.append(dim_h)
            dim_w = MultiArrayDimension()
            dim_w.label = 'width'
            dim_w.size = occ_map.shape[1]
            msg.layout.dim.append(dim_w)
            msg.data = occ_map.flatten().tolist()
            self.occ_publisher.publish(msg)
            return msg
        

    def sim(self, trial_idx=0, init_queue=None):

        start_time = time.time()
        cfg = self.cfg

        # get important params
        steps = cfg.sim.steps
        t_max = cfg.sim.t_max if cfg.sim.t_max else np.inf
        horizon = cfg.a_star.horizon
        replan = cfg.a_star.replan
        seed = cfg.get('seed', None)
        dt = cfg.controller.dt

        # setup pymunk environment
        space = pymunk.Space()  # threaded=True causes some issues
        space.iterations = cfg.sim.iterations
        space.gravity = cfg.sim.gravity
        space.damping = cfg.sim.damping

        # keep track of running total of total kinetic energy / total impulse
        # computed using pymunk api call, source code here
        # https://github.com/slembcke/Chipmunk2D/blob/edf83e5603c5a0a104996bd816fca6d3facedd6a/src/cpArbiter.c#L158-L172
        # https://www.pymunk.org/en/latest/pymunk.html#pymunk.Arbiter.total_ke
                                # source code in Chimpunk2D cpArbiterTotalKE
        total_ke = [0, []]  # keep track of both running total and ke at each collision
        total_impulse = [0, []]
        # keep track of running total of work
        total_work = [0, []]

        total_dis = 0 
        prev_state = None   

        # keep track of all the obstacles that collide with ship
        clln_obs = set()

        # keep track of contact points
        contact_pts = []

        # setup pymunk collision callbacks
        def pre_solve_handler(arbiter, space, data):
            ice_body = arbiter.shapes[1].body
            return True

        def post_solve_handler(arbiter, space, data):
            nonlocal total_ke, total_impulse, clln_obs
            ship_shape, ice_shape = arbiter.shapes

            total_ke[0] += arbiter.total_ke
            total_ke[1].append(arbiter.total_ke)

            total_impulse[0] += arbiter.total_impulse.length
            total_impulse[1].append(list(arbiter.total_impulse))

            if arbiter.is_first_contact:
                clln_obs.add(arbiter.shapes[1])

            # max of two sets of points, easy to see with a picture with two overlapping convex shapes
            # find the impact locations in the local coordinates of the ship
            for i in arbiter.contact_point_set.points:
                contact_pts.append(list(arbiter.shapes[0].body.world_to_local((i.point_b + i.point_a) / 2)))

        handler = space.add_collision_handler(1, 2)
        # from pymunk docs
        # post_solve: two shapes are touching and collision response processed
        handler.pre_solve = pre_solve_handler
        handler.post_solve = post_solve_handler

        start = init_queue['ship_state']
        obs_dicts = init_queue['obstacles']
        currents = init_queue['currents'] if 'currents' in init_queue else None
        
        # filter out obstacles that have zero area
        obs_dicts[:] = [ob for ob in obs_dicts if poly_area(ob['vertices']) != 0]
        obs_vertices = [ob['vertices'] for ob in obs_dicts]
        thicknesses = [ob['thickness'] for ob in obs_dicts]

        # we don't case about goal_x; goal_y + 1 to overshot the planner a bit for avoiding stopping early
        goal = (0, cfg.goal_y + 1)
        
        polygons = generate_sim_obs(space, obs_dicts, cfg.sim.obstacle_density)
        for p in polygons:
            p.collision_type = 2
            
        # send ship pose
        ship_state_msg = Pose2D()
        ship_state_msg.x = float(start[0])
        ship_state_msg.y = float(start[1])
        ship_state_msg.theta = float(start[2])
        self.ship_pose_publisher.publish(ship_state_msg)

        # send goal
        goal_msg = Point()
        goal_msg.x = float(goal[0])
        goal_msg.y = float(goal[1])
        self.goal_publisher.publish(goal_msg)

        # send trial idx
        trial_idx_msg = Int32()
        trial_idx_msg.data = trial_idx
        self.trial_idx_publisher.publish(trial_idx_msg)

        # clear current path to receive new path
        self.new_path = None
        self.path = None
        self.new_path_received = False

        # initialize ship sim objects
        ship_body, ship_shape = Ship.sim(cfg.ship.vertices, start)
        ship_shape.collision_type = 1
        space.add(ship_body, ship_shape)
        
        # run initial simulation steps to let environment settle
        for _ in range(200):
            space.step(dt / steps)
        
        if currents:
            apply_currents(polygons, currents)
            
        velocities = [np.asarray(p.body.velocity) for p in polygons]
        msg = self.publush_planner_msg(obs_vertices, thicknesses, velocities, trial_idx=trial_idx)
        
        prev_obs = CostMap.get_obs_from_poly(polygons)

        # Wait for the first path. Keeping publishing while waiting
        while rclpy.ok() and (self.new_path is None):
            self.trial_idx_publisher.publish(trial_idx_msg)
            if cfg.planner == 'predictive' or cfg.planner == 'icenet':
                self.occ_publisher.publish(msg)
            else:
                self.poly_publisher.publish(msg)
            self.ship_pose_publisher.publish(ship_state_msg)
            self.goal_publisher.publish(goal_msg)
            self.wait_path_rate.sleep()

        self.path = np.copy(self.new_path[self.new_path[:, 1] < horizon + self.new_path[0, 1]])
        self.new_path_received = False

        # setup dp controller
        cx = self.path.T[0]
        cy = self.path.T[1]
        ch = self.path.T[2]
        dp = DP(x=start[0], y=start[1], yaw=start[2],
                cx=cx, cy=cy, ch=ch, output_dir=self.output_dir,
                **cfg.controller)
        state = dp.state

        
        plot = Plot(
            np.zeros((cfg.costmap.m, cfg.costmap.n)), obs_dicts, path=self.path.T,
            ship_pos=start, ship_vertices=np.asarray(ship_shape.get_vertices()),
            horizon=horizon, map_figsize=None, y_axis_limit=cfg.plot.y_axis_limit,
            target=tuple(dp.setpoint[:2]), inf_stream=True, goal=goal[1], currents=currents,
        ) 

        ship_state = ([], [])  # keep track of ship path
        past_path = ([], [])  # keep track of planned path behind ship
        t = 0  # start time tick
        goal_op = ge if not cfg.get('reverse_dir') else le

        try:
            work = 0.0

            if cfg.anim.init_save:
                plot.animate_sim(save_fig_dir=os.path.join(self.output_dir, 't' + str(trial_idx)), suffix=t, im_format=cfg.anim.format)
            
            # main simulation loop
            while t < t_max:
                t += 1
                if t >= t_max:
                    print('Reached max time: ', t_max)
                    break

                if currents:
                    apply_currents([*polygons, ship_shape], currents)

                if goal_op(ship_body.position.y, cfg.goal_y):
                    break

                if t % cfg.plan_steps == 0:
                    if not cfg.collection:
                        print(f'Simulation time {t} / {t_max}, ship position x={ship_body.position.x} y={ship_body.position.y}', end='\r')

                    # get updated obstacles
                    obs_vertices = CostMap.get_obs_from_poly(polygons)
                    velocities = [np.asarray(p.body.velocity) for p in polygons]
                
                    # update work metric
                    work = total_work_done(prev_obs, obs_vertices)
                    total_work[0] += work
                    total_work[1].append(work)
                    prev_obs = obs_vertices
                    
                    if cfg.collection and t > cfg.skip_start:
                        metrics = [total_ke[0],]
                        self.occupancy.save_snapshot(obs_vertices, thicknesses, velocities,
                                                     ship_body, ship_shape.get_vertices(), metrics)

                    if replan:
                        # send new information for replan
                        # send trial idx
                        trial_idx_msg = Int32()
                        trial_idx_msg.data = trial_idx
                        self.trial_idx_publisher.publish(trial_idx_msg)
                        
                        msg = self.publush_planner_msg(obs_vertices, thicknesses, velocities, trial_idx=trial_idx, t=t)

                        # send ship pose
                        ship_state_msg = Pose2D()
                        ship_state_msg.x = state.x
                        ship_state_msg.y = state.y
                        ship_state_msg.theta = state.yaw
                        self.ship_pose_publisher.publish(ship_state_msg)

                    # check for path
                    if self.new_path_received:
                        # confirm path is a minimum of 2 points
                        if len(self.new_path) > 1:
                            self.path = np.copy(self.new_path)
                            cx = self.path.T[0]
                            cy = self.path.T[1]
                            ch = self.path.T[2]
                            dp.target_course.update(cx, cy, ch)
                        self.new_path_received = False

                if self.cfg.planner == 'skeleton':
                    # update DP controller
                    dp(ship_body.position.x,
                    ship_body.position.y,
                    ship_body.angle)

                    # apply velocity commands to ship body
                    ship_body.angular_velocity = state.r * np.pi / 180
                    x_vel, y_vel = state.get_global_velocity()  # get velocities in global frame
                    ship_body.velocity = Vec2d(x_vel, y_vel)

                else:
                    # call ideal controller
                    omega, global_velocity = dp.ideal_control(ship_body.position.x,
                    ship_body.position.y,
                    ship_body.angle)

                    # apply velocity commands to ship body from ideal controller
                    ship_body.angular_velocity = omega
                    ship_body.velocity = Vec2d(global_velocity[0], global_velocity[1])
                
                # move simulation forward
                for _ in range(steps):
                    space.step(dt / steps)

                # update ship pose
                state.update_pose(ship_body.position.x,
                                ship_body.position.y,
                                ship_body.angle)

                ship_state[0].append(state.x)
                ship_state[1].append(state.y)

                if prev_state is not None:
                    dis = ((state.x - prev_state[0])**2 + (state.y - prev_state[1])**2)**(0.5)
                    total_dis += dis
                prev_state = [state.x, state.y]

                # log updates including tracking error
                (e_x, e_y, e_yaw), track_idx = tracking_error([state.x, state.y, state.yaw], self.path, get_idx=True)
                past_path[0].append(self.path[track_idx][0])
                past_path[1].append(self.path[track_idx][1])

                # update setpoint
                x_s, y_s, h_s = dp.get_setpoint()
                dp.setpoint = np.asarray([x_s, y_s, np.unwrap([state.yaw, h_s])[1]])

                if t % cfg.anim.plot_steps == 0 and cfg.anim.save:
                    plt.pause(0.001)

                    # update animation
                    plot.update_path(self.path[track_idx:].T, target=(x_s, y_s), ship_state=ship_state,
                                    past_path=past_path, start_y=self.path[0, 1])
                    plot.update_ship(ship_body, ship_shape, move_yaxis_threshold=cfg.anim.move_yaxis_threshold)
                    # get updated obstacles
                    plot.update_obstacles(obstacles=CostMap.get_obs_from_poly(polygons))
                    if cfg.anim.title:
                        plot.sim_fig.suptitle('velocity ({:.2f}, {:.2f}, {:.2f}) [m/s, m/s, rad/s]'
                                            '\nsim iteration {:d}'
                                            .format(ship_body.velocity.x, ship_body.velocity.y, ship_body.angular_velocity, t), x=0.6)
                    plot.animate_sim(save_fig_dir=os.path.join(self.output_dir, 't' + str(trial_idx))
                                    if self.output_dir else None, suffix=t, im_format=cfg.anim.format)

                # frequency control to ensure the simulation does not exceed real-world time
                # this is a quick work-around to ensure the navigation node and planner node are on the same time scale
                if not cfg.collection:
                    self.rate_dt.sleep()

        finally:
            sim_time = time.time() - start_time
            travel_time = t * dt
            print(f'Sim time: {sim_time}')
            if cfg.collection:
                print('Exporting data')
                self.occupancy.export(os.path.join(self.output_dir, f't{trial_idx}'), data_size=cfg.data_size)
            self.occupancy.clear_data()
            print('Done simulation\nCleaning up...')
            print('Total KE', total_ke[0])
            print('Total impulse', total_impulse[0])
            print('Total work {}'.format(total_work[0]))
            print('Total distance: ', total_dis)
            print(f'Travel time: {travel_time}')
            plt.ioff()
            plt.close('all')

        return  total_ke[0], total_impulse[0], total_work[0], total_dis, sim_time, travel_time

    def run_sim(self):

        if self.conc not in [0.2, 0.3, 0.4, 0.5]:
            raise Exception("Invalid concentration value. Please check the config file. ") 

        ddict = pickle.load(open(self.env_file, 'rb'))
        print("Planner: ", self.cfg.planner, "; Concentration: ", self.conc)

        ke_list = []
        impulse_list = []
        work_list = []
        dis_list = []
        sim_time_list = []
        travel_time_list = []
        print(self.trial_range)
        for trial_idx in range(*self.trial_range):
            exp = ddict['exp'][self.conc][trial_idx] 

            total_ke, total_impulse, total_work, total_dis, sim_time, travel_time = self.sim(
                trial_idx=trial_idx,
                init_queue={
                    **exp
                })
            
            if not self.cfg.collection:
                ke_list.append(total_ke)
                impulse_list.append(total_impulse)
                work_list.append(total_work)
                dis_list.append(total_dis)
                sim_time_list.append(sim_time)
                travel_time_list.append(travel_time)

                metrics_dict = {
                    "ke": np.array(ke_list),
                    "impulse": np.array(impulse_list),
                    "work": np.array(work_list),
                    "dis": np.array(dis_list),
                    "sim_time": np.array(sim_time_list),
                    "travel_time": np.array(travel_time_list)
                }
                np.save(os.path.join(self.output_dir, f"{int(self.conc*100)}_{self.cfg.planner}_metrics.npy"), metrics_dict)
        # Send finished simulation message by setting trial idx to -1
        trial_idx_msg = Int32()
        trial_idx_msg.data = -1
        self.trial_idx_publisher.publish(trial_idx_msg)
        print("Finished!")

if __name__ == '__main__':
    rclpy.init(args=None)
    
    parser = argparse.ArgumentParser(description="Run the simulation.")
    parser.add_argument('--cfg', type=str, required=True, help="Path to the configuration file.")
    parser.add_argument('--env', type=str, required=True, help="Path to the environment file.")
    parser.add_argument('--dir', type=str, default='outputs', help="Output directory for the simulation results.")
    parser.add_argument('--conc', type=int, default=20, choices=[20, 30, 40, 50], required=True, help="Concentration of ice in percentage (20, 30, 40, or 50).")
    parser.add_argument('--trial_range', type=int, nargs=2, required=True, help="Trial range as two integers, e.g., 0 50.")
    parser.add_argument('--id', type=str, default='1', help="Instance id of the run to create a unique communication channel.")

    args = parser.parse_args()
    
    cfg_file = args.cfg
    cfg = DotDict.load_from_file(cfg_file)
    conc =  args.conc / 100
    
    node = Sim2DNode(cfg=cfg, env_file=args.env, output_dir=args.dir, conc=conc, trial_range=args.trial_range, id=args.id)

    thread = threading.Thread(target=rclpy.spin, args=(node, ), daemon=True)
    thread.start()

    try:
        node.run_sim()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
