import numpy as np
import cv2, imageio
import os

import skimage
from scipy.spatial.transform import Rotation as R
import navigation.validity_func.depth_utils as du
import navigation.validity_func.validity_utils as vu

import matplotlib
# matplotlib.use('tkagg')
import matplotlib.pyplot as plt



def build_mapper(args):
    params = {}
    camera_height = args.sensor_height
    map_size_cm = args.map_size_cm
    params["frame_width"] = args.front_width
    params["frame_height"] = args.front_height
    params["fov"] = args.front_hfov
    params["resolution"] = args.map_resolution
    params["map_size_cm"] = map_size_cm
    params["agent_min_z"] = 25
    params["agent_medium_z"] = 100
    params["agent_max_z"] = 150
    params["agent_height"] = camera_height * 100
    params["agent_view_angle"] = np.pi
    params["du_scale"] = 1
    params["vision_range"] = 64
    params["use_mapper"] = 1
    params["visualize"] = 0
    params["maze_task"] = 1
    params["obs_threshold"] = 1
    mapper = MapBuilder(params)
    return mapper


class MapBuilder(object):
    def __init__(self, params):
        self.params = params
        frame_width = params["frame_width"]
        frame_height = params["frame_height"]
        fov = params["fov"]
        self.camera_matrix = du.get_camera_matrix(frame_width, frame_height, fov)
        self.vision_range = params["vision_range"]
        self.map_size_cm = params["map_size_cm"]
        self.resolution = params["resolution"]
        agent_min_z = params["agent_min_z"]
        agent_medium_z = params["agent_medium_z"]
        agent_max_z = params["agent_max_z"]
        self.z_bins = [agent_min_z, agent_medium_z, agent_max_z]
        self.du_scale = params["du_scale"]
        self.use_mapper = params["use_mapper"]
        self.visualize = params["visualize"]
        self.maze_task = params["maze_task"]
        self.obs_threshold = params["obs_threshold"]

        self.est_map = np.zeros(
            (
                self.map_size_cm // self.resolution + 1,
                self.map_size_cm // self.resolution + 1,
                len(self.z_bins) + 1,
            ),
            dtype=np.float32,
        )

        self.gt_map = np.zeros(
            (
                self.map_size_cm // self.resolution + 1,
                self.map_size_cm // self.resolution + 1,
                len(self.z_bins) + 1,
            ),
            dtype=np.float32,
        )

        self.agent_height = params["agent_height"]
        self.agent_view_angle = params["agent_view_angle"]

        self.prev_pose = np.array([0, 0, 0])

    def update_map(self, depth, current_pose, gt=False, agent_view=None):
        if gt:
            curr_map = self.gt_map
        else:
            curr_map = self.est_map

        if agent_view is None:
            mask2 = depth > 9999.0
            depth[mask2] = 0.0

            for i in range(depth.shape[1]):
                depth[:, i][depth[:, i] == 0.0] = depth[:, i].max()

            mask1 = depth == 0
            depth[mask1] = np.NaN

            point_cloud = du.get_point_cloud_from_z(
                depth, self.camera_matrix, scale=self.du_scale
            )

            agent_view = du.transform_camera_view(
                point_cloud, self.agent_height, self.agent_view_angle
            )

        # shift_loc = [self.vision_range * self.resolution // 2, 0, np.pi / 2.0]
        # agent_view_centered = du.transform_pose(agent_view, shift_loc)
        #
        # agent_view_flat, is_valids = du.bin_points(
        #     agent_view_centered, self.vision_range, self.z_bins, self.resolution
        # )
        #
        # x1 = self.map_size_cm // (self.resolution * 2) - self.vision_range // 2
        # x2 = x1 + self.vision_range
        # y1 = self.map_size_cm // (self.resolution * 2)
        # y2 = y1 + self.vision_range
        # agent_view_cropped = agent_view_flat[:, :, 1] + agent_view_flat[:, :, 2]
        #
        # agent_view_cropped = agent_view_cropped / self.obs_threshold
        # agent_view_cropped[agent_view_cropped >= 0.5] = 1.0
        # agent_view_cropped[agent_view_cropped < 0.5] = 0.0
        #
        # agent_view_explored = agent_view_flat.sum(2)
        # agent_view_explored[agent_view_explored > 0] = 1.0

        geocentric_pc = du.transform_pose(agent_view, current_pose)

        geocentric_flat, is_valids = du.bin_points(
            geocentric_pc, curr_map.shape[0], self.z_bins, self.resolution
        )

        curr_map = curr_map + geocentric_flat

        map_gt = (curr_map[:, :, 1] + curr_map[:, :, 2]) // self.obs_threshold
        map_gt[map_gt >= 0.5] = 1.0
        map_gt[map_gt < 0.5] = 0.0

        wall_map_gt = curr_map[:, :, 2] // self.obs_threshold
        map_gt[map_gt >= 0.5] = 1.0
        map_gt[map_gt < 0.5] = 0.0

        explored_gt = curr_map.sum(2)
        explored_gt[explored_gt > 1] = 1.0

        if gt:
            self.gt_map = curr_map
        else:
            self.est_map = curr_map

        return map_gt, explored_gt, wall_map_gt

    def get_curr_obsmap(self, depth):
        curr_map = self.gt_map

        mask2 = depth > 9999.0
        depth[mask2] = 0.0

        for i in range(depth.shape[1]):
            depth[:, i][depth[:, i] == 0.0] = depth[:, i].max()

        mask1 = depth == 0
        depth[mask1] = np.NaN

        point_cloud = du.get_point_cloud_from_z(
            depth, self.camera_matrix, scale=self.du_scale
        )

        agent_view = du.transform_camera_view(
            point_cloud, self.agent_height, self.agent_view_angle
        )

        geocentric_pc = du.transform_pose(agent_view, (self.map_size_cm/2, self.map_size_cm/2, 0))

        geocentric_flat, is_valids = du.bin_points(
            geocentric_pc, curr_map.shape[0], self.z_bins, self.resolution
        )


        curr_map = geocentric_flat

        map_gt = (curr_map[:, :, 1] + curr_map[:, :, 2]) // self.obs_threshold
        map_gt[map_gt >= 0.5] = 1.0
        map_gt[map_gt < 0.5] = 0.0

        wall_map_gt = curr_map[:, :, 2] // self.obs_threshold
        map_gt[map_gt >= 0.5] = 1.0
        map_gt[map_gt < 0.5] = 0.0

        explored_gt = curr_map.sum(2)
        explored_gt[explored_gt > 1] = 1.0

        return agent_view, map_gt, explored_gt, wall_map_gt


    def get_st_pose(self, current_loc):
        loc = [
            -(
                current_loc[0] / self.resolution
                - self.map_size_cm // (self.resolution * 2)
            )
            / (self.map_size_cm // (self.resolution * 2)),
            -(
                current_loc[1] / self.resolution
                - self.map_size_cm // (self.resolution * 2)
            )
            / (self.map_size_cm // (self.resolution * 2)),
            90 - np.rad2deg(current_loc[2]),
        ]
        return loc

    def reset_map(self):
        self.est_map = np.zeros(
            (
                self.map_size_cm // self.resolution + 1,
                self.map_size_cm // self.resolution + 1,
                len(self.z_bins) + 1,
            ),
            dtype=np.float32,
        )

        self.gt_map = np.zeros(
            (
                self.map_size_cm // self.resolution + 1,
                self.map_size_cm // self.resolution + 1,
                len(self.z_bins) + 1,
            ),
            dtype=np.float32,
        )

        self.prev_pose = np.array([0, 0, 0])

    def get_map(self, gt=False):
        if gt:
            return self.gt_map
        else:
            return self.est_map

    def get_mapper_pose_from_sim_pose(self, sim_pose, sim_origin):
        x, y, o = vu.get_rel_pose_change(sim_pose, sim_origin)
        return (
            self.map_size_cm - (x * 100.0 + self.map_size_cm / 2.0),
            self.map_size_cm - (y * 100.0 + self.map_size_cm / 2.0),
            o,
        )

    def get_map_grid_from_sim_pose_cm(self, sim_pose):
        return (
            int(sim_pose[0] / self.resolution),
            int(sim_pose[1] / self.resolution),
        )

    def get_sim_pose_from_mapper_coords(self, mapper_coords, cur_center_position, cur_rotation):
        """
        Input:
            mapper_coords: mapper coordinates (x, y)
            mapper_size: (l, w)
            sim_origin: 3-dof sim_pose which cooresponds to
                        (l/2, w/2) in mapper coordinates
        Output:
            sim_pose:  3-dof sim pose
        """
        map_resolution = self.resolution
        mapper_size = self.map_size_cm
        center = [mapper_size / map_resolution / 2.0, mapper_size / map_resolution / 2.0]

        dx = (mapper_coords[0] - center[0]) * map_resolution / 100.
        dy = (mapper_coords[1] - center[1]) * map_resolution / 100.

        rot = R.from_rotvec(cur_rotation)
        rot.as_matrix()
        delta = rot.apply(np.array([-dx, 0, -dy]))

        target_position = np.array([cur_center_position[0] + delta[0], cur_center_position[1] + delta[1],
                                    cur_center_position[2] + delta[2]])
        return target_position
        # pos1 = [
        #     mapper_size * map_resolution / 200.0,
        #     mapper_size * map_resolution / 200.0,
        #     0.0,
        # ]

        # dx = (mapper_coords[0] - mapper_size / 2.0) * map_resolution / 100.0
        # dy = (mapper_coords[1] - mapper_size / 2.0) * map_resolution / 100.0
        #
        # pos2 = [
        #     mapper_size * map_resolution / 200.0 - dy,
        #     mapper_size * map_resolution / 200.0 - dx,
        #     0.0,
        # ]
        #
        # rel_pose_change = vu.get_rel_pose_change(pos2, pos1)

        # sim_pose = vu.get_new_pose(sim_origin, rel_pose_change)
        # return np.array(sim_pose)

    def is_traversable(self, grid_map, start, end, agent_size=4):
        """
        Check if a path between two points on a 2D grid map is traversable by an agent of size d.

        Args:
            gridmap (ndarray): A 2D numpy array representing the grid map, where 0 indicates
                               a free space and 1 indicates an occupied space.
            start (tuple): A tuple representing the starting point as (row, col).
            end (tuple): A tuple representing the ending point as (row, col).
            d (int): The size of the agent in pixels.

        Returns:
            bool: True if the path is traversable, False otherwise.
        """

        x1, y1 = start
        x2, y2 = end
        dx = x2 - x1
        dy = y2 - y1

        if dx == 0 and dy == 0:
            return True

        steps = max(abs(dx), abs(dy))
        half_agent_size = agent_size // 2

        for i in range(1, steps + 1):
            x = int(round(x1 + i * dx / steps))
            y = int(round(y1 + i * dy / steps))

            for j in range(-half_agent_size, half_agent_size + 1):
                for k in range(-half_agent_size, half_agent_size + 1):
                    if (
                            0 <= y + j < len(grid_map)
                            and 0 <= x + k < len(grid_map[0])
                            and grid_map[x + k][y + j] > 0
                    ):
                        return False

        return True

