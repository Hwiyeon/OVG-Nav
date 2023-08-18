import os
os.environ["OMP_NUM_THREADS"] = "1"
import quaternion
import numpy as np
import cv2
import skimage
import seaborn as sns
from navigation.validity_func.fmm_planner import FMMPlanner
from navigation.validity_func.map_builder import build_mapper
from navigation.validity_func.validity_utils import (
    get_l2_distance,
    get_sim_location,
    get_rel_pose_change,
)
# from navigation.navigation_utils.sim_utils import NoisySensor


class LocalAgent(object):
    def __init__(
        self,
        args
    ):
        self.args = args
        self.actuation_noise = args.noisy_action
        # self.pose_noise = args.noisy_pose
        self.noisy_sensor = None
        # if self.actuation_noise or self.pose_noise:
        #     self.noisy_sensor = NoisySensor(args, noise_level=1.0)
        self.mapper = build_mapper(args)
        self.map_size_cm = args.map_size_cm
        self.map_resolution = args.map_resolution
        self.collision = False
        self.gt_collision_locs = []
        self.est_collision_locs = []

        self.action_idx_map = ['stop', 'move_forward', 'turn_left', 'turn_right']

        self.edge_range = self.args.edge_range
        self.max_edge_range = self.args.edge_range * 1.5
        self.grid_m = self.mapper.params['resolution'] * 0.01



    def reset_with_curr_pose(self, curr_pos, curr_rot, set_rot_bias=False):
        ## initialize the local map
        self.curr_pos = curr_pos
        self.curr_rot = curr_rot
        self.sim_origin = get_sim_location(
            self.curr_pos, quaternion.from_rotation_vector(self.curr_rot)
        )
        self.initialize_local_map_pose()
        if set_rot_bias:
            self.o_gt_bias = self.curr_rot[1]
            self.o_est_bias = self.curr_rot[1]
        self.stg_x, self.stg_y = int(self.y_gt / self.map_resolution), int(
            self.x_gt / self.map_resolution
        )
        self.est_new_sim_origin = self.sim_origin
        self.gt_new_sim_origin = self.sim_origin
        self.reset_goal = True
        self.gt_collision_locs = []
        self.est_collision_locs = []


    def initialize_local_map_pose(self):
        self.mapper.reset_map()
        self.x_gt, self.y_gt, self.o_gt = (
            self.map_size_cm / 2.0,
            self.map_size_cm / 2.0,
            0.0,
        )
        self.x_est, self.y_est, self.o_est = (
            self.map_size_cm / 2.0,
            self.map_size_cm / 2.0,
            0.0,
        )

        self.o_gt_bias = 0.0
        self.o_est_bias = 0.0

        self.reset_goal = True
        self.sim_origin = get_sim_location(
            self.curr_pos, quaternion.from_rotation_vector(self.curr_rot)
        )

        self.est_local_visit = np.zeros(self.mapper.get_map()[:, :, 1].shape)
        self.gt_local_visit = np.zeros(self.mapper.get_map()[:, :, 1].shape)
        self.mapper.prev_pose = (self.x_gt, self.y_gt, self.o_gt)

    def update_curr_pose(self, curr_pos, curr_rot):
        self.curr_pos = curr_pos
        self.curr_rot = curr_rot

    def update_gt_local_map(self, depth_m, agent_view=None):
        x_gt, y_gt, o_gt = self.get_mapper_pose_from_sim_pose(
            self.gt_new_sim_origin,
            self.sim_origin,
        )
        self.update_visited_map([self.x_gt, self.y_gt], [x_gt, y_gt], gt=True)

        self.x_gt, self.y_gt, self.o_gt = x_gt, y_gt, o_gt
        x, y, o = self.x_gt, self.y_gt, self.o_gt

        depth_cm = depth_m * 100.
        self.gt_local_map, self.gt_local_exp_map, _ = self.mapper.update_map(
            depth_cm, (x, y, o),
            # curr_depth_img, (x, y, o)
            gt=True,
            agent_view = agent_view
        )
        self.mapper.prev_pose = (x, y, o)
        for locs in self.gt_collision_locs:
            self.gt_local_map[locs[0], locs[1]] = 1.0
            self.gt_local_exp_map[locs[0], locs[1]] = 1.0
        if self.collision:
            if [self.stg_x, self.stg_y] not in self.gt_collision_locs:
                self.gt_collision_locs.append([self.stg_x, self.stg_y])
                self.gt_local_map[self.stg_x, self.stg_y] = 1.0
                self.gt_local_exp_map[self.stg_x, self.stg_y] = 1.0
            self.collision = False

            # self.gt_local_map[int(y / self.map_resolution), int(x / self.map_resolution)] = 1.0
            # self.gt_local_exp_map[int(y / self.map_resolution), int(x / self.map_resolution)] = 1.0
            # self.gt_local_map[self.stg_x, self.stg_y] = 1.0
            # self.gt_local_exp_map[self.stg_x, self.stg_y] = 1.0
            # self.gt_local_map[max(0, self.stg_x-1):min(self.gt_local_map.shape[0]-1, self.stg_x+1),
            #                     max(0, self.stg_y-1):min(self.gt_local_map.shape[1]-1, self.stg_y+1)] = 1.0
            # self.gt_local_exp_map[max(0, self.stg_x-1):min(self.gt_local_map.shape[0]-1, self.stg_x+1),
            #                     max(0, self.stg_y-1):min(self.gt_local_map.shape[1]-1, self.stg_y+1)] = 1.0


    def update_est_local_map(self, depth_m):
        x_est, y_est, o_est = self.get_mapper_pose_from_sim_pose(
            self.est_new_sim_origin,
            self.sim_origin,
        )
        self.update_visited_map([self.x_est, self.y_est], [x_est, y_est], gt=False)

        self.x_est, self.y_est, self.o_est = x_est, y_est, o_est
        x, y, o = self.x_est, self.y_est, self.o_est

        depth_cm = depth_m * 100.
        self.est_local_map, self.est_local_exp_map, _ = self.mapper.update_map(
            depth_cm, (x, y, o)
            # curr_depth_img, (x, y, o)
        )
        for locs in self.est_collision_locs:
            self.est_local_map[locs[0], locs[1]] = 1.0
            self.est_local_exp_map[locs[0], locs[1]] = 1.0
        if self.collision:
            if [self.stg_x, self.stg_y] not in self.est_collision_locs:
                self.est_collision_locs.append([self.stg_x, self.stg_y])
                self.est_local_map[self.stg_x, self.stg_y] = 1.0
                self.est_local_exp_map[self.stg_x, self.stg_y] = 1.0
            self.collision = False

            # self.est_local_map[int(y / self.map_resolution), int(x / self.map_resolution)] = 1.0
            # self.est_local_exp_map[int(y / self.map_resolution), int(x / self.map_resolution)] = 1.0
            # self.est_local_map[self.stg_x, self.stg_y] = 1.0
            # self.est_local_exp_map[self.stg_x, self.stg_y] = 1.0
            # self.est_local_map[max(0, self.stg_x - 1):min(self.est_local_map.shape[0] - 1, self.stg_x + 1),
            #                     max(0, self.stg_y - 1):min(self.est_local_map.shape[1] - 1, self.stg_y + 1)] = 1.0
            # self.est_local_exp_map[max(0, self.stg_x - 1):min(self.est_local_map.shape[0] - 1, self.stg_x + 1),
            #                     max(0, self.stg_y - 1):min(self.est_local_map.shape[1] - 1, self.stg_y + 1)] = 1.0




    def get_mapper_pose_from_sim_pose(self, sim_pose, sim_origin):
        x, y, o = get_rel_pose_change(sim_pose, sim_origin)
        return (
            self.map_size_cm - (x * 100.0 + self.map_size_cm / 2.0),
            self.map_size_cm - (y * 100.0 + self.map_size_cm / 2.0),
            o,
        )

    def set_goal(self, delta_dist, delta_rot):
        start = (
            int(self.y_gt / self.map_resolution),
            int(self.x_gt / self.map_resolution),
        )
        goal = (
            start[0]
            + int(
                delta_dist * np.sin(delta_rot + self.o_gt + self.o_gt_bias) * 100.0 / self.map_resolution
            ),
            start[1]
            + int(
                delta_dist * np.cos(delta_rot + self.o_gt + self.o_gt_bias) * 100.0 / self.map_resolution
            ),
        )
        self.goal = goal


    def navigate_local(self, gt=False):
        if gt:
            local_map = self.gt_local_map
            x, y, o = self.x_gt, self.y_gt, self.o_gt
        else:
            local_map = self.est_local_map
            x, y, o = self.x_est, self.y_est, self.o_est
        traversible = (
            skimage.morphology.binary_dilation(
                local_map, skimage.morphology.disk(2)
            )
            != True
        )

        start = (
            int(y / self.map_resolution),
            int(x / self.map_resolution),
        )

        try:
            traversible[start[0] - 2 : start[0] + 3, start[1] - 2 : start[1] + 3] = 1
        except:
            import ipdb

            ipdb.set_trace()

        for locs in self.gt_collision_locs:
            traversible[locs[0], locs[1]] = 0

        planner = FMMPlanner(self.args, traversible, 360 // 10, 1)

        if self.reset_goal:
            planner.set_goal(self.goal, auto_improve=True)
            self.goal = planner.get_goal()
            self.reset_goal = False
        else:
            planner.set_goal(self.goal, auto_improve=True)

        stg_x, stg_y = start
        stg_x, stg_y, replan = planner.get_short_term_goal2((stg_x, stg_y))

        # if get_l2_distance(start[0], self.goal[0], start[1], self.goal[1]) < 3:
        # if planner.fmm_dist[start[0], start[1]] > np.max(planner.fmm_dist) * 0.9: ## not navigable
        if planner.fmm_dist[start[0], start[1]] > self.max_edge_range / self.grid_m * 1.5 or \
            planner.fmm_dist[start[0], start[1]] > np.max(planner.fmm_dist) * 0.9:  ## not directly navigable
            terminate = 1
        else:
            terminate = 0

        agent_orientation = np.rad2deg(o)
        action = planner.get_next_action(start, (stg_x, stg_y), agent_orientation)
        self.stg_x, self.stg_y = int(stg_x), int(stg_y)
        return action, terminate

    def get_map(self, gt=False):
        if gt:
            local_map, local_exp_map = self.gt_local_map, self.gt_local_exp_map
        else:
            local_map, local_exp_map = self.est_local_map, self.est_local_exp_map


        self.stg_x, self.stg_y = int(self.y_gt / self.map_resolution), int(
            self.x_gt / self.map_resolution
        )
        metric_map = local_map + 0.5 * local_exp_map * 255
        metric_map[
            int(self.stg_x) - 1 : int(self.stg_x) + 1,
            int(self.stg_y) - 1 : int(self.stg_y) + 1,
        ] = 255
        metric_map = cv2.resize(metric_map / 255.0, (80, 80))

        # metric_map = self.local_map + 0.5 * self.local_exp_map * 255
        # metric_map = metric_map.astype("uint8")
        # metric_map = cv2.cvtColor(metric_map, cv2.COLOR_GRAY2RGB)
        # self.stg_x, self.stg_y = int(self.y_gt / self.map_resolution), int(
        #     self.x_gt / self.map_resolution
        # )
        # metric_map[
        #     int(self.stg_x) - 1 : int(self.stg_x) + 1,
        #     int(self.stg_y) - 1 : int(self.stg_y) + 1,
        #     :,
        # ] = [
        #     0,
        #     0,
        #     255,
        # ]
        # metric_map = cv2.resize(metric_map, (80, 80))
        # metric_map = metric_map.astype("float")
        # metric_map /= 255
        ##
        # metric_map = torch.from_numpy(metric_map)
        return metric_map

    def update_visited_map(self, last_start, start, gt=False):
        last_start = [
            int(last_start[0] / self.map_resolution),
            int(last_start[1] / self.map_resolution),
        ]
        start = [
            int(start[0] / self.map_resolution),
            int(start[1] / self.map_resolution),
        ]

        steps = 25
        for i in range(steps):
            y = int(last_start[0] + (start[0] - last_start[0]) * (i + 1) / steps)
            x = int(last_start[1] + (start[1] - last_start[1]) * (i + 1) / steps)
            # x = np.shape(self.local_visit)[1] - int(last_start[1] + (start[1] - last_start[1]) * (i + 1) / steps)
            if not gt:
                self.est_local_visit[x, y] = 1
            else:
                self.gt_local_visit[x, y] = 1

    def get_neareset_navigable_goal(self, grid_map, start, goal):
        """
        Get neareest navigable goal from the given goal.

        Args:
            gridmap (ndarray): A 2D numpy array representing the grid map, where 0 indicates
                               a free space and 1 indicates an occupied space.
            start (tuple): A tuple representing the starting point as (row, col).
            end (tuple): A tuple representing the ending point as (row, col).
            d (int): The size of the agent in pixels.

        Returns:
            bool: True if the path is traversable, False otherwise.
        """

        traversible = (skimage.morphology.binary_dilation(grid_map, skimage.morphology.disk(2)) != True)

        try:
            traversible[start[0] - 2: start[0] + 3, start[1] - 2: start[1] + 3] = 1
        except:
            import ipdb
            ipdb.set_trace()
        planner = FMMPlanner(self.args, traversible, 360 // 10, 1)
        planner.set_goal(start, auto_improve=True)

        grid_map = planner.fmm_dist
        map_size = grid_map.shape[0]
        fmm_threshold = np.max(grid_map) * 0.9
        search_size = 5 * 4


        x1, y1 = start
        x2, y2 = goal
        dx = x2 - x1
        dy = y2 - y1

        steps = max(abs(dx), abs(dy))
        goal_updated = False
        if steps == 0:
            return goal, goal_updated

        for i in range(0, steps + 1):
            x = int(round(x2 - i * dx / steps))
            y = int(round(y2 - i * dy / steps))

            if 0 <= y < len(grid_map) and 0 <= x < len(grid_map[0]) and grid_map[x][y] < fmm_threshold:
                if i > 0:
                    goal_updated = True
                    near_goal_map = grid_map[max(0, x - search_size):min(map_size-1, x + search_size),
                                    max(0, y - search_size):min(map_size-1, y + search_size)]
                    near_goal_map = np.logical_and(near_goal_map - grid_map[x][y] < search_size, near_goal_map - grid_map[x][y] > -search_size)
                    row_id, col_id = np.where(near_goal_map)
                    dist_to_goal = fmm_threshold
                    gx, gy = x, y
                    for i in range(len(row_id)):
                        temp_dist = np.linalg.norm(np.array([row_id[i]+max(0, x - search_size), col_id[i]+ max(0, y - search_size)]) - goal)
                        if temp_dist < dist_to_goal:
                            gx, gy = row_id[i]+max(0, x - search_size), col_id[i]+ max(0, y - search_size)
                            dist_to_goal = temp_dist
                    x, y = gx, gy


                return (x, y), goal_updated

        return goal, goal_updated


    def get_observed_colored_map(self, gt=True):

        def fill_color(colored, mat, color):
            for i in range(3):
                colored[:, :, 2 - i] *= (1 - mat)
                colored[:, :, 2 - i] += (1 - color[i]) * mat
            return colored

        goal = self.goal
        gt_map = self.gt_local_map
        gt_visited = self.gt_local_visit

        if gt:
            obs_map = self.gt_local_map
            explored = self.gt_local_exp_map
            visited = self.gt_local_visit
            x, y, o = self.x_gt, self.y_gt, self.o_gt
        else:
            obs_map = self.est_local_map
            explored = self.est_local_exp_map
            visited = self.est_local_visit
            x, y, o = self.x_est, self.y_est, self.o_est


        m, n = obs_map.shape
        colored = np.zeros((m, n, 3))
        pal = sns.color_palette("Paired")

        if gt_map is not None:
            current_palette = [(0.8, 0.8, 0.8)]
            colored = fill_color(colored, gt_map, current_palette[0])

        current_palette = [(235. / 255., 243. / 255., 1.)]
        colored = fill_color(colored, explored, current_palette[0])

        green_palette = sns.light_palette("green")
        colored = fill_color(colored, obs_map, pal[2])
        if gt_map is not None:
            current_palette = [(0.6, 0.6, 0.6)]
            colored = fill_color(colored, obs_map * gt_map, pal[3])

        current_palette = [(0.9, 0.9, 0.9)]
        map_explored = explored * obs_map
        colored = fill_color(colored, explored, current_palette[0])
        colored = fill_color(colored, obs_map * map_explored, pal[3])

        red_palette = sns.light_palette("red")
        if gt_visited is not None:
            colored = fill_color(colored, gt_visited, current_palette[0])
        colored = fill_color(colored, visited, pal[4])
        if gt_visited is not None:
            colored = fill_color(colored, visited * gt_visited, pal[5])

        current_palette = sns.color_palette()

        selem = skimage.morphology.disk(4)
        goal_mat = np.zeros((m, n))
        goal_mat[goal[0], goal[1]] = 1
        goal_mat = 1 - skimage.morphology.binary_dilation(
            goal_mat, selem) != True

        colored = fill_color(colored, goal_mat, current_palette[0])

        current_palette = sns.color_palette("Paired")

        x, y = int(x / 5.0), int(y / 5.0)

        # agent_mat = np.zeros((m, n))
        # agent_mat[x, y] = 1
        # agent_mat = 1 - skimage.morphology.binary_dilation(
        #     agent_mat, selem) != True
        #
        # colored = fill_color(colored, agent_mat, current_palette[4])

        colored = 1 - colored
        colored *= 255
        colored = colored.astype(np.uint8)





        agent_size = 8
        fc = (128,128,128)
        dx = np.cos(o)
        dy = np.sin(o)

        arrow_mag = 3

        # start_x = int(np.round(x - dx * agent_size /2))
        # start_y = int(np.round(y - dy * agent_size /2))
        start_x = x
        start_y = y
        end_x = int(np.round(x + dx * agent_size))
        end_y = int(np.round(y + dy * agent_size))

        # Draw the main line of the arrow
        colored = cv2.line(colored, (start_x, start_y), (end_x, end_y), fc, thickness=2)

        # Now draw the arrowhead
        angle = np.arctan2(y - end_y, x - end_x)
        p1_x = int(end_x + arrow_mag * np.cos(angle + np.pi / 4))
        p1_y = int(end_y + arrow_mag * np.sin(angle + np.pi / 4))
        colored = cv2.line(colored, (end_x, end_y), (p1_x, p1_y), fc, thickness=2)

        p2_x = int(end_x + arrow_mag * np.cos(angle - np.pi / 4))
        p2_y = int(end_y + arrow_mag * np.sin(angle - np.pi / 4))
        colored = cv2.line(colored, (end_x, end_y), (p2_x, p2_y), fc, thickness=2)

        return colored

