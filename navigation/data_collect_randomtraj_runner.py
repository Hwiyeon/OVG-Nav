# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import math
import multiprocessing
import os
os.environ["OMP_NUM_THREADS"] = "1"
import random
import time
from enum import Enum
import cv2
import torch
torch.set_num_threads(1)

import numpy as np
from PIL import Image
# from settings import default_sim_settings, make_cfg
from navigation.configs.settings_pano_navi import make_cfg
from habitat_sim.utils.common import d3_40_colors_rgb
# from detector.detector_mask import Detector
import quaternion
from scipy.spatial.transform import Rotation as R
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
import skimage
# from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
# from habitat.utils.visualizations import maps

import habitat_sim
import habitat_sim.agent
from habitat_sim import bindings as hsim
from habitat_sim.physics import MotionType
from habitat_sim.utils.common import (
    d3_40_colors_rgb,
    download_and_unzip,
    quat_from_angle_axis,
)


_barrier = None

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as patches
import time
import json
from utils.visualizations.maps import get_topdown_map_from_sim, to_grid, TopdownView
from utils.graph_utils.graph_pano_cs import GraphMap
from utils.obj_category_info import assign_room_category, obj_names_det as obj_names, gibson_goal_obj_names, mp3d_goal_obj_names, room_names, mp3d_room_names

from tqdm import tqdm
import pickle

from modules.detector.detector_mask import Detector
from modules.free_space_model.inference import FreeSpaceModel
from modules.comet_relation.inference import CommonSenseModel
from modules.visual_odometry.keypoint_matching import KeypointMatching
from navigation.local_navigation import LocalNavigation

from navigation.validity_func.map_builder import build_mapper
from navigation.validity_func.fmm_planner import FMMPlanner
from validity_func.local_nav import LocalAgent
from validity_func.validity_utils import (
    get_relative_location,
    get_sim_location,
)

class DemoRunnerType(Enum):
    BENCHMARK = 1
    EXAMPLE = 2
    AB_TEST = 3

def cuboid_data(center, size):
    """
       Create a data array for cuboid plotting.


       ============= ================================================
       Argument      Description
       ============= ================================================
       center        center of the cuboid, triple
       size          size of the cuboid, triple, (x_length,y_width,z_height)
       :type size: tuple, numpy.array, list
       :param size: size of the cuboid, triple, (x_length,y_width,z_height)
       :type center: tuple, numpy.array, list
       :param center: center of the cuboid, triple, (x,y,z)


      """


    # suppose axis direction: x: to left; y: to inside; z: to upper
    # get the (left, outside, bottom) point
    o = [a - b / 2 for a, b in zip(center, size)]
    # get the length, width, and height
    l, w, h = size
    x = [[o[0], o[0] + l, o[0] + l, o[0], o[0]],  # x coordinate of points in bottom surface
         [o[0], o[0] + l, o[0] + l, o[0], o[0]],  # x coordinate of points in upper surface
         [o[0], o[0] + l, o[0] + l, o[0], o[0]],  # x coordinate of points in outside surface
         [o[0], o[0] + l, o[0] + l, o[0], o[0]]]  # x coordinate of points in inside surface
    y = [[o[1], o[1], o[1] + w, o[1] + w, o[1]],  # y coordinate of points in bottom surface
         [o[1], o[1], o[1] + w, o[1] + w, o[1]],  # y coordinate of points in upper surface
         [o[1], o[1], o[1], o[1], o[1]],          # y coordinate of points in outside surface
         [o[1] + w, o[1] + w, o[1] + w, o[1] + w, o[1] + w]]    # y coordinate of points in inside surface
    z = [[o[2], o[2], o[2], o[2], o[2]],                        # z coordinate of points in bottom surface
         [o[2] + h, o[2] + h, o[2] + h, o[2] + h, o[2] + h],    # z coordinate of points in upper surface
         [o[2], o[2], o[2] + h, o[2] + h, o[2]],                # z coordinate of points in outside surface
         [o[2], o[2], o[2] + h, o[2] + h, o[2]]]                # z coordinate of points in inside surface
    return np.array(x), np.array(y), np.array(z)


class Runner:
    def __init__(self, args, sim_settings, COI, dataset, data_type='val'):
        self.args = args
        self._sim_settings = sim_settings.copy()
        self.dataset = dataset
        self.dataset_info = None
        self.gt_planner = None
        self.data_type = data_type

        self.det_COI = COI
        self.obj_names = obj_names
        if args.dataset == 'mp3d':
            self.goal_obj_names = mp3d_goal_obj_names
        elif args.dataset == 'gibson':
            self.goal_obj_names = gibson_goal_obj_names
        # self.sge_th = args.sge_th
        self.pix_num = args.width*args.height
        self.cand_angle = np.arange(-90, 270, args.cand_rot)
        self.cand_angle_bias = list(self.cand_angle).index(0) # 0 degree is the center
        self.edge_range = args.edge_range
        self.last_mile_range = args.last_mile_range

        self.vo_height = args.front_height
        self.vo_width = args.front_width
        self.vo_hfov = args.front_hfov

        self.height = args.height
        self.width = args.width

        self.pano_width = args.pano_width
        self.pano_height = args.pano_height

        self.hfov = args.hfov

        self.step_size = args.move_forward
        self.follower_goal_radius = 0.75 * self.step_size
        self.act_rot = args.act_rot
        self.cand_rot_angle = args.cand_rot
        self.rot_num = len(self.cand_angle)
        self.max_trial = 100
        self.max_step = args.max_step

        # self.vo_pred_model = VO_prediction(args.vo_config)
        self.depth_scale = np.iinfo(np.uint16).max

        # self.detector = Detector(args, self.det_COI)
        # self.free_space_model = FreeSpaceModel(args)
        self.common_sense_model = CommonSenseModel(args)
        self.vo_model = KeypointMatching(args)

        self.local_navi_module = LocalNavigation(self.args, self.vo_model)
        self.local_agent = LocalAgent(self.args)
        self.local_mapper = build_mapper(self.args)

        self.vis_floorplan = args.vis_floorplan
        self.use_oracle = args.use_oracle



    def save_rgbd_video(self, rgb_list, depth_list, save_dir, env_name, idx, panoramic=False):

        data_dir = f"{save_dir}/{self.data_type}/{env_name}/{env_name}_{idx:04d}"
        if not os.path.exists(data_dir): os.makedirs(data_dir)

        if panoramic:
            width = self.pano_width
            height = self.pano_height
            rgb_name = 'pano_rgb'
            depth_name = 'pano_depth'
        else:
            width = self.vo_width
            height = self.vo_height
            rgb_name = 'rgb'
            depth_name = 'depth'

        video = cv2.VideoWriter(f'{data_dir}/{rgb_name}.avi', cv2.VideoWriter_fourcc(*'XVID'), 5,
                                (width, height))
        for image in rgb_list:
            image = cv2.cvtColor((image[:, :, :3] / 255.).astype(np.float32), cv2.COLOR_RGB2BGR)
            video.write((image * 255).astype(np.uint8))
        video.release()

        video = cv2.VideoWriter(f'{data_dir}/{depth_name}.avi', cv2.VideoWriter_fourcc(*'XVID'), 5,
                                (width, height), isColor=False)
        for depth_obs in depth_list:
            # norm_depth = np.where(depth_obs < 10, depth_obs/10., 1.).astype(np.float32)
            # norm_depth = (norm_depth * np.iinfo(np.uint16).max).astype(np.uint16)
            depth_obs = (np.clip(depth_obs, 0.1, 10.) / 10.).astype(np.float32)
            depth_obs = (depth_obs * self.depth_scale).astype(np.uint16) / self.depth_scale
            depth_obs = (depth_obs * 255).astype(np.uint8)
            video.write(depth_obs)
        video.release()

    def save_video(self, frame_list, save_dir, env_name, idx):
        data_dir = f"{save_dir}/{self.data_type}/{env_name}/{env_name}_{idx:04d}"
        if not os.path.exists(data_dir): os.makedirs(data_dir)

        width = np.shape(frame_list[0])[1]
        height = np.shape(frame_list[0])[0]

        video = cv2.VideoWriter(f'{data_dir}/graph.avi', cv2.VideoWriter_fourcc(*'XVID'), 5,
                                (width, height))
        for image in frame_list:
            image = cv2.cvtColor((image[:, :, :3] / 255.).astype(np.float32), cv2.COLOR_RGB2BGR)
            video.write((image * 255).astype(np.uint8))
        video.release()

    def make_total_frame(self, rgb, depth, graph, info):
        rh, rw = np.shape(rgb)[:2]
        rh, rw = int(rh/2), int(rw/2)
        small_rgb = cv2.resize(rgb, (rw, rh))
        small_depth = cv2.resize(depth, (rw, rh))
        small_depth = ((np.clip(small_depth, 0.1, 10.) / 10.) * 255).astype(np.uint8)
        gh, gw = np.shape(graph)[:2]
        gh, gw = rh*2, int(rh*2 * gw / gh)

        small_graph = cv2.resize(graph, (gw, gh))
        max_h = max(rh*2, gh)

        frame = np.zeros([max_h, rw+gw, 3])
        frame[:rh, :rw, :] = small_rgb[:, :, :3]
        frame[rh:, :rw, :] = np.tile(small_depth[:, :, np.newaxis], [1, 1, 3])
        frame[:gh, rw:, ] = small_graph
        frame = frame.astype(np.uint8)


        ## text
        text1 = "Target object goal: {}   Mode: {}".format(info['target_goal'], info['mode'])
        text2 = "Position: {}".format(info['cur_position'])
        font_color = (255, 255, 255)
        text_size, _ = cv2.getTextSize(text1, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        text_position1 = (int((frame.shape[1] - text_size[0]) / 2), frame.shape[0] + text_size[1] * 2 + 10)
        text_position2 = (int((frame.shape[1] - text_size[0]) / 2), frame.shape[0] + text_size[1] * 2 + 25)
        canvas_height = frame.shape[0] + text_size[1] * 2 + 40
        canvas_width = frame.shape[1]
        canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
        canvas[:frame.shape[0], :] = frame
        # canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        frame2 = cv2.putText(canvas, text1, text_position1, cv2.FONT_HERSHEY_SIMPLEX, 0.4, font_color, 1, cv2.LINE_AA)
        frame2 = cv2.putText(canvas, text2, text_position2, cv2.FONT_HERSHEY_SIMPLEX, 0.4, font_color, 1, cv2.LINE_AA)

        return frame2
        # figure, ax = plt.subplots(1, 1, facecolor="whitesmoke")
        # figure.show()
        # figure.canvas.draw()
        #
        # ax.clear()
        #
        # def draw(ax, img):
        #     ax.imshow(img)
        #     ax.set_yticks([])
        #     ax.set_xticks([])
        #     ax.set_yticklabels([])
        #     ax.set_xticklabels([])
        #     ax.text(0.5, -0.1, 'Target object goal: {}   Mode: {}\n Position: {}'.format(info['target_goal'],
        #                                                                                      info['mode'],
        #                                                                                      info['cur_position']),
        #                 ha='center', fontsize=10, transform=ax.transAxes)
        #     for _ in range(5):
        #         plt.tight_layout()
        #     return ax
        #
        # background = figure.canvas.copy_from_bbox(ax.bbox)
        # figure.canvas.restore_region(background)
        # ax.draw_artist(draw(ax, frame))
        #
        #
        #
        #
        #
        #
        # # for _ in range(5):
        # #     plt.tight_layout()
        # # figure.canvas.draw()
        # out_img = np.fromstring(figure.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        # out_img = out_img.reshape(figure.canvas.get_width_height()[::-1] + (3,))
        # plt.close()
        #
        # return out_img


    def panoramic_obs(self, obs, semantic=False):

        rgb_panoramic = np.zeros([self.pano_height, self.pano_width, 3]).astype(int)
        depth_panoramic = np.zeros([self.pano_height, self.pano_width])
        if semantic:
            semantic_panoramic = np.zeros([self.pano_height, self.pano_width]).astype(int)

        for i, rot in enumerate([ '270', '0', '90', '180']):

            rgb_panoramic[:, i * self.width:(i + 1) * self.width, :] = obs[f'rgb_{rot}'][:, :, :3]   # 320 - 320* np.tan(30/180*np.pi) / np.tan(35/180*np.pi) = 56
            depth_panoramic[:, i * self.width:(i + 1) * self.width] = obs[f'depth_{rot}']

        return {
            'rgb_panoramic': rgb_panoramic,
            'depth_panoramic': depth_panoramic
        }


    def init_random(self):

        self._cfg = make_cfg(self._sim_settings)
        self._sim = habitat_sim.Simulator(self._cfg)

        random.seed(self._sim_settings["seed"])
        self._sim.seed(self._sim_settings["seed"])

    def random_inital_traj_agent(self, min_dist=0.5, max_dist=None):
        # initialize the agent at a random start state
        agent_id = self._sim_settings["default_agent"]
        agent = self._sim.initialize_agent(agent_id)
        start_state = agent.get_state()

        start_state.position = self._sim.pathfinder.get_random_navigable_point()
        level_idx = -1
        for i, level in enumerate(self.level_range):
            if start_state.position[1] > level[0] and start_state.position[1] < level[1]:
                level_idx = i
                break
        if level_idx == -1:
            raise ValueError("starting position not in any level")

        start_state.sensor_states = dict()
        agent.set_state(start_state)

        start_position = start_state.position
        # force starting position on first floor (try 100 samples)
        trial = 0
        not_valid = True
        ged_dist = np.inf
        while not_valid and trial < 100:
            trial += 1
            goal_position = self._sim.pathfinder.get_random_navigable_point()
            if not goal_position[1] > self.level_range[level_idx][0] and goal_position[1] < self.level_range[level_idx][1]:
                continue

            path = habitat_sim.ShortestPath()
            path.requested_start = start_position
            path.requested_end = goal_position

            try:
                path_action = self.follower.find_path(goal_position)
                if len(path_action) > 500:
                    continue
            except:
                continue

            if not max_dist is None:
                if self._sim.pathfinder.find_path(path) and path.geodesic_distance > max_dist:
                    continue
            if self._sim.pathfinder.find_path(path) and path.geodesic_distance < min_dist:
                continue

            if self._sim.pathfinder.find_path(path) and abs(start_position - goal_position)[1] < 0.5:
                not_valid = False
                ged_dist = path.geodesic_distance


        return start_state, goal_position, not_valid, ged_dist


    def init_commonsense_candidate_room(self, goal_names, candidate_names):
        goal_category_room = {}
        goal_category_room_feat = {}
        goal_category_room_score = {}

        # cand_category_room = {}
        # cand_category_room_feat = {}
        # cand_category_room_score = {}
        # cand_room_feat = self.common_sense_model.clip.get_text_feat(candidate_names).type(torch.float32)
        for i, goal_name in enumerate(goal_names):
            pred_words = self.common_sense_model.gen_pred_words(gibson_goal_obj_names[i], num_generate=20)
            # pred_words = pred_words[0]
            pred_words_feat = self.common_sense_model.clip.get_text_feat(pred_words).type(torch.float32)

            goal_category_room[goal_name] = pred_words
            goal_category_room_feat[goal_name] = pred_words_feat.cpu()
            goal_category_room_score[goal_name] = np.ones_like(pred_words).astype(float)

            # value, indice = torch.max(self.common_sense_model.clip.get_sim_from_feats(cand_room_feat, pred_words_feat, normalize=True).type(torch.float32),
            #                           dim=0)
            # topk = torch.topk(value, k=len(candidate_names), largest=True).indices.detach().cpu().numpy()
            # cand_category_room[goal_name] = [candidate_names[i] for i in topk]
            # cand_category_room_feat[goal_name] = torch.stack([cand_room_feat[i].detach().cpu() for i in topk])
            # cand_category_room_score[goal_name] = np.array([value[i].detach().cpu().numpy() for i in topk])

        return goal_category_room, goal_category_room_feat, goal_category_room_score

    def calculate_navmesh(self):
        navmesh_settings = habitat_sim.NavMeshSettings()
        navmesh_settings.set_defaults()
        navmesh_success = self._sim.recompute_navmesh(self._sim.pathfinder, navmesh_settings, include_static_objects=True)
        print("navmesh_success ", navmesh_success )
        self.tdv = TopdownView(self._sim, self.args.dataset, data_dir=self.args.floorplan_data_dir)
        self.scene_height = self._sim.agents[0].state.position[1]
        self.tdv.draw_top_down_map(height=self.scene_height)
        # self.curr_rot = self.tdv.get_polar_angle(quaternion.from_rotation_vector(self.abs_init_rotation))



    def get_topdown_floorplan(self):
        if abs(self._sim.agents[0].state.position[1] - self.scene_height) > 0.5:
            self.scene_height = self._sim.agents[0].state.position[1]
            self.tdv.draw_top_down_map(height=self.scene_height)
        # self.tdv.draw_agent()
        # if len(self.map_paths) > 0:
        #     tdv = self.tdv.draw_paths(self.map_paths, tdv)
        return self.tdv.rgb_top_down_map

    def update_cur_floor_map(self):
        # self.map = get_topdown_map_from_sim(self._sim, meters_per_pixel=0.02)
        # self.map_size = np.shape(self.map)
        self.map = self.get_topdown_floorplan()
        self.map = cv2.cvtColor(self.map, cv2.COLOR_BGR2RGB)
        self.map_size = np.shape(self.map)



    def get_vis_grid_pose(self, pose):
        if len(pose) == 3:
            pose = np.array(pose) + np.array(self.abs_init_position)
            grid_y, grid_x = to_grid(pose[2], pose[0], self.map_size, self._sim)
        elif len(pose) == 2:
            grid_y, grid_x = to_grid(pose[1], pose[0], self.map_size, self._sim)
        return (grid_x, grid_y)

    def node_value_by_obj_dist(self, dist, max_dist=15.0):
        return max(1 - dist / max_dist, 0)

    def vis_topdown_graph_map(self, vis_map, graph_map, vis_obj_score=None, curr_node_id=None, curr_goal_node_id=None, curr_goal_position=None):
        node_list = list(graph_map.node_by_id.values())

        for edge in list(graph_map.edges):
            if not edge.draw:
                node_grid1 = self.get_vis_grid_pose(edge.nodes[0].pos)
                node_grid2 = self.get_vis_grid_pose(edge.nodes[1].pos)
                vis_map = cv2.line(vis_map, node_grid1, node_grid2, (0, 64, 64), 5)
                edge.draw = True



        for node in node_list:

            # if node.draw and node.nodeid != curr_node_id and node.nodeid != curr_goal_node_id:
            #     continue

            node_grid = self.get_vis_grid_pose(node.pos)
            if vis_obj_score is not None:
                # color = (np.array((0, 255, 0)) * self.node_value_by_obj_dist(node.dist_to_objs[vis_obj_score])).astype(int)
                color = (np.array((0, 255, 0)) * node.cm_score).astype(int)
                color = tuple([color[i].item() for i in range(3)])
                cand_color = (np.array((0, 0, 255)) * self.node_value_by_obj_dist(
                    node.dist_to_objs[vis_obj_score])).astype(int)
                cand_color = tuple([cand_color[i].item() for i in range(3)])
                goal_color = (255, 255, 0)
            else:
                color = (0, 255, 0)
                cand_color = (0, 0, 255)
                goal_color = (255, 255, 0)
            if node.visited:
                # if node.is_start:
                if node.nodeid == curr_node_id:
                    vis_map = cv2.circle(vis_map, node_grid, 10, (255,0,0), -1)
                else:
                    vis_map = cv2.circle(vis_map, node_grid, 10, color, -1)
            elif node.nodeid == curr_goal_node_id:
                vis_map = cv2.circle(vis_map, node_grid, 10, goal_color, -1)
            else:
                vis_map = cv2.circle(vis_map, node_grid, 10, cand_color, -1)

            node.draw = True

        if curr_goal_position is not None:
            node_grid = self.get_vis_grid_pose(curr_goal_position)
            vis_map = cv2.circle(vis_map, node_grid, 10, (255, 0, 255), -1)


        return vis_map

    def vis_pos_on_topdown_map(self, pos):
        vis_map = self.map.copy()
        node_grid = self.get_vis_grid_pose(pos)
        vis_map = cv2.circle(vis_map, node_grid, 10, (0, 255, 0), -1)
        return vis_map

    def set_level_range(self):
        self.level_range = []
        for level in self._sim.semantic_scene.levels:
            self.level_range.append([
                level.aabb.center[1] - level.aabb.sizes[1] / 2,
                level.aabb.center[1] + level.aabb.sizes[1] / 2,
            ])

        return

    def get_bbox_from_pos_size(self, position, size):
        bbox = np.array([
            [position[0] - size[0] / 2, position[2] - size[2] / 2],
            [position[0] + size[0] / 2, position[2] + size[2] / 2]
        ])
        grid_bbox = np.array([
            self.get_vis_grid_pose(bbox[0]),
            self.get_vis_grid_pose(bbox[1])
        ])
        return grid_bbox

    def vis_topdown_obj_map(self, vis_map, obj_category):
        for obj in self.env_obj_info:
            agent_level = self.check_position2level(self.scene_height)
            if obj['category'] == obj_category and agent_level == obj['level']:
                grid_bbox = self.get_bbox_from_pos_size(obj['position'], obj['sizes'])
                grid_bbox = grid_bbox.tolist()
                shapes = np.zeros_like(vis_map, np.uint8)
                cv2.rectangle(shapes, tuple(grid_bbox[0]), tuple(grid_bbox[1]), (0, 0, 128), -1)
                alpha = 0.3
                mask = shapes.astype(bool)
                vis_map[mask] = cv2.addWeighted(vis_map, alpha, shapes, 1 - alpha, 0)[mask]

        return vis_map

    def vis_topdown_goal_obj_map(self, vis_map, goal_obj_category):
        for obj in self.env_goal_obj_info:
            agent_level = self.check_position2level(self.scene_height)
            if obj['category'] == goal_obj_category and agent_level == obj['level']:
                grid_bbox = self.get_bbox_from_pos_size(obj['position'], obj['sizes'])
                grid_bbox = grid_bbox.tolist()
                shapes = np.zeros_like(vis_map, np.uint8)
                cv2.rectangle(shapes, tuple(grid_bbox[0]), tuple(grid_bbox[1]), (255, 0, 0), -1)
                alpha = 0.3
                mask = shapes.astype(bool)
                vis_map[mask] = cv2.addWeighted(vis_map, alpha, shapes, 1-alpha, 0)[mask]
                # vis_map = cv2.rectangle(vis_map, tuple(grid_bbox[0]), tuple(grid_bbox[1]), (255, 0, 0), -1)
        return vis_map


    def vis_topdown_map_with_captions(self, graph_map, curr_node=None, curr_goal_node=None, curr_goal_position=None, vis_goal_obj_score=None, vis_obj=None, save_dir=None):
        # vis_map = self.map.copy()
        vis_map = self.cur_map
        if vis_obj is not None:
            vis_map = self.vis_topdown_obj_map(vis_map, vis_obj)
        if vis_goal_obj_score is not None:
            vis_map = self.vis_topdown_goal_obj_map(vis_map, vis_goal_obj_score)

        curr_node_id, curr_goal_node_id = None, None
        if curr_node is not None:
            curr_node_id = curr_node.nodeid
        if curr_goal_node is not None:
            curr_goal_node_id = curr_goal_node.nodeid
        vis_map = self.vis_topdown_graph_map(vis_map, graph_map, vis_goal_obj_score,
                                                curr_node_id=curr_node_id, curr_goal_node_id=curr_goal_node_id,
                                                curr_goal_position=curr_goal_position)

        return vis_map
        # figure, ax = plt.subplots(1, 1, facecolor="whitesmoke")
        # figure.show()
        # figure.canvas.draw()
        #
        # ax.clear()
        #
        # def draw(ax, img):
        #     ax.imshow(img)
        #     ax.set_yticks([])
        #     ax.set_xticks([])
        #     ax.set_yticklabels([])
        #     ax.set_xticklabels([])
        #     for _ in range(5):
        #         plt.tight_layout()
        #     return ax
        #
        # background = figure.canvas.copy_from_bbox(ax.bbox)
        # figure.canvas.restore_region(background)
        # ax.draw_artist(draw(ax, vis_map))
        # figure.canvas.blit(ax.bbox)
        #
        # # ax.set_title(f'Target object goal : {goal_obj_names[vis_goal_obj_score]}\n', wrap=True, horizontalalignment='center', fontsize=12)
        #
        #
        # if vis_obj is not None:
        #     plt.figtext(0.5, 0.05, f'Object position : {obj_names[vis_obj]}\n', wrap=True, horizontalalignment='center', fontsize=8)
        # if save_dir is not None:
        #     plt.savefig(save_dir)
        # else:
        #
        #     out_img = np.fromstring(figure.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        #     out_img = out_img.reshape(figure.canvas.get_width_height()[::-1] + (3,))
        #     plt.close()
        #     return out_img
        # plt.close()
        # return None


    def check_position2room(self, position, room_info, vis=False):

        ## y = height

        cur_room = []
        find_room = False

        if vis:
            fig = plt.figure()
            ax = fig.gca(projection='3d')
            ax.set_xlabel('X')
            ax.set_xlim(-10, 10)
            ax.set_ylabel('Y')
            ax.set_ylim(-10, 10)
            ax.set_zlabel('Z')
            ax.set_zlim(-10, 10)

            ax.scatter(position[0], position[1], position[2], color='r')
            for room in room_info:
                X, Y, Z = cuboid_data(room['center'], room['sizes'])
                ax.plot_surface(X, Y, Z, color='b', rstride=1, cstride=1, alpha=0.1)

            plt.show()

        for i, room in enumerate(room_info):
            if position[0] < room['center'][0] + (room['sizes'][0] / 2.) and \
                    position[0] > room['center'][0] - (room['sizes'][0] / 2.) and \
                    position[1] < room['center'][1] + (room['sizes'][1] / 2.) and \
                    position[1] > room['center'][1] - (room['sizes'][1] / 2.) and \
                    position[2] < room['center'][2] + (room['sizes'][2] / 2.) and \
                    position[2] > room['center'][2] - (room['sizes'][2] / 2.):
                find_room = True
                cur_room.append(room['category'])

        return cur_room

    def check_position2level(self, pos):
        # if len(position) == 3:
        #     pos = position[1]
        # else:
        #     pos = position

        for i, level in enumerate(self.level_range):
            if pos > level[0] and pos < level[1]:
                return str(i)

    def check_goal_point_validity(self, start_position, goal_position, is_goal_obj=False):

        path = habitat_sim.ShortestPath()
        # path = habitat_sim.MultiGoalShortestPath()
        path.requested_start = start_position
        path.requested_end = goal_position

        if not is_goal_obj:
            is_valid_point = self._sim.pathfinder.is_navigable(goal_position)
        else:
            is_valid_point = True
        is_valid_path = self._sim.pathfinder.find_path(path)
        is_valid = is_valid_point and is_valid_path and not path.geodesic_distance == np.inf

        # if path.geodesic_distance == np.inf:
        #     is_valid = False

        return is_valid, path.geodesic_distance

    def get_geodesic_distance_to_object_category(self, position, category, target_view_points=None):
        path = habitat_sim.MultiGoalShortestPath()
        path.requested_start = np.array(position)
        if target_view_points is None:
            path.requested_ends = self.env_class_goal_view_point[category]
        else:
            path.requested_ends = target_view_points
        self._sim.pathfinder.find_path(path)
        return path.geodesic_distance

    def dist_euclidean_floor(self, pos1, pos2):
        return np.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[2] - pos2[2]) ** 2)

    def dist_to_objs(self, pos):
        dist = np.full(len(self.goal_obj_names), np.inf)
        is_valid_point = True

        # for i in range(len(self.goal_obj_names)):
        dist[self.goal_class_idx] = self.get_geodesic_distance_to_object_category(pos, self.goal_info['category'])



        # cur_goal_info = self._sim.semantic_scene.objects[int(self.goal_info['id'].split('_')[-1])]
        # for i, obj in enumerate(self.env_goal_obj_info):
        #
        #     is_valid, geo_dist = self.check_goal_point_validity(pos, obj['position'], is_goal_obj=True)
        #     if not is_valid: pass
        #
        #     if geo_dist < dist[obj['category']]:
        #         dist[obj['category']] = geo_dist
        #         if obj['category'] == self.goal_class_idx:
        #             cur_goal_info = self._sim.semantic_scene.objects[obj['id']]


                    # self.update_goal_info(goal_info)


        if np.sum(dist != np.full(len(self.goal_obj_names), np.inf)) == 0:
            is_valid_point = False
        # self.update_goal_info(cur_goal_info)

        return dist, is_valid_point


    def find_nearest_goal(self, pos, goal_class_idx):
        dist = np.inf
        # cur_goal_info = self._sim.semantic_scene.objects[int(self.goal_info['id'].split('_')[-1])]
        goal_idx = 0
        for i, obj in enumerate(self.env_goal_obj_info):
            if obj['category'] == goal_class_idx:
                is_valid, geo_dist = self.check_goal_point_validity(pos, obj['position'], is_goal_obj=True)
                if not is_valid: pass
                if geo_dist < dist:
                    dist = geo_dist
                    goal_idx = i

        return self.env_goal_obj_info[goal_idx]


    def update_goal_info(self, goal_info):
        if self.args.dataset == 'mp3d':
            if int(goal_info.id.split('_')[-1]) in self.goal_id_to_viewpoints:
                view_points = [v['agent_state']['position'] for v in self.goal_id_to_viewpoints[int(goal_info.id.split('_')[-1])]]
                path = habitat_sim.MultiGoalShortestPath()
                path.requested_start = np.array(self.abs_init_position)
                path.requested_ends = view_points
                self._sim.pathfinder.find_path(path)
                if len(path.points) > 0:
                    best_point = path.points[-1]
                else:
                    best_point = None
        else:
            best_point = None
            view_points = None
        # view_points = None

        return {'position': goal_info.aabb.center, 'sizes': goal_info.aabb.sizes, 'category': goal_info.category.name(),
                'id': goal_info.id,
                'best_viewpoint_position': best_point,
                'view_points': np.array(view_points)}


    def get_dirc_imgs_from_pano(self, pano_img, num_imgs=12):
        pw, ph = self.pano_width, self.pano_height

        # split the panorama into 12 square images with even angles
        dirc_imgs = []
        for i in range(num_imgs):
            angle = i * 360 / num_imgs
            x = int(pw * (angle / 360))
            dirc_img = pano_img[:, x:x + ph]
            if x + ph > pw:
                dirc_img = np.concatenate((dirc_img, pano_img[:, :x + ph - pw]), axis=1)
            dirc_imgs.append(dirc_img)
        #         print(np.shape(dirc_img))
        return np.array(dirc_imgs)

    def get_cand_node(self, pano_rgb, pos, head, goal_info):

        cand_nodes = []

        text = goal_info['category_place']

        rot_axis = np.array([0, 1, 0])
        # head = -quaternion.as_rotation_vector(rot)[1] * 180 / np.pi

        # for global coordinate
        # turn left = positive angle
        # free cand angle idx --> right side is positive
        free_cand_nodes = self.free_space_model.predict_free_space(pano_rgb)

        for i, angle in enumerate(self.cand_angle):
            if free_cand_nodes[i] == 0: continue
            rot_vec = head + np.radians(-angle) * rot_axis
            unit_vec = -np.array([np.sin(rot_vec[1]), 0, np.cos(rot_vec[1])])
            # unit_vec = np.array([np.sin(rot_vec[1]), 0, -np.cos(rot_vec[1])])

            cand_pos = pos + unit_vec * self.edge_range
            cand_rot = quaternion.from_rotation_vector(rot_vec * np.pi / 180)

            cand_nodes.append({'position': cand_pos, 'rotation': cand_rot})

        pano_split_images = self.get_dirc_imgs_from_pano(pano_rgb)
        cand_split_images = pano_split_images[np.where(free_cand_nodes == 1)[0]]

        torch.set_num_threads(1)
        similarity, cand_split_feat = self.common_sense_model.clip.get_text_image_sim(text, cand_split_images,
                                                                                      out_img_feat=True)
        value = np.round(np.max(similarity, axis=1), 3)

        for i in range(len(cand_nodes)):
            cand_nodes[i]['clip_feat'] = cand_split_feat[i]
            cand_nodes[i]['value'] = value[i]

        return cand_nodes

    def get_cand_node_dirc(self, pano_rgb, depth, pos, rot):
        ## rot is rotation vector
        cur_heading_idx = int(np.round(-rot[1] * 180 / np.pi / self.cand_rot_angle)) % self.rot_num
        cand_nodes = []
        cand_angle = [-30, 0, 30]
        self.local_mapper.reset_map()
        depth_cm = depth * 100
        pose_origin_for_map = (pos[0], pos[2], 0)  # (x, y, o)
        pose_for_map = (pos[0], pos[2], rot[1])  # (x, y, o)
        pose_on_map_cm = self.local_mapper.get_mapper_pose_from_sim_pose(pose_for_map, pose_origin_for_map)
        pose_on_map = self.local_mapper.get_map_grid_from_sim_pose_cm(pose_on_map_cm)

        ### get current local map ###
        curr_local_map, curr_exp_map, _ = self.local_mapper.update_map(depth_cm, pose_on_map_cm)
        # text = goal_info['category_place']
        rot_axis = np.array([0, 1, 0])
        # head = -quaternion.as_rotation_vector(rot)[1] * 180 / np.pi

        # for global coordinate
        # turn left = positive angle
        # free cand angle idx --> right side is positive
        free_cand_nodes = np.zeros(12)
        angle_bias = np.where(self.cand_angle == -30)[0][0]

        for i, angle in enumerate(cand_angle):
            rot_vec = rot + np.radians(-angle) * rot_axis
            unit_vec = -np.array([np.sin(rot_vec[1]), 0, np.cos(rot_vec[1])])
            cand_pos = pos + unit_vec * self.edge_range
            cand_rot = rot_vec
            cur_heading_idx = int(np.round(-rot_vec[1] * 180 / np.pi / self.cand_rot_angle)) % self.rot_num

            ## map coordinate for checking free space
            cand_pose_for_map = (cand_pos[0], cand_pos[2], rot_vec[1])
            cand_pose_on_grid_map_cm = self.local_mapper.get_mapper_pose_from_sim_pose(cand_pose_for_map, pose_origin_for_map)
            cand_pose_on_grid_map = self.local_mapper.get_map_grid_from_sim_pose_cm(cand_pose_on_grid_map_cm)
            if self.local_mapper.is_traversable(curr_local_map, pose_on_map, cand_pose_on_grid_map):
                cand_node_info = {'position': cand_pos, 'rotation': cand_rot, 'heading_idx': cur_heading_idx}


                # for degbugging vis
                # vis_map = np.copy(curr_local_map)
                # vis_map[pose_on_map[0], pose_on_map[1]] = 2
                # vis_map[cand_pose_on_grid_map[0], cand_pose_on_grid_map[1]] = 2
                # plt.imsave('test_map.png',vis_map, origin='lower')

                # ## --- one step further node --- ##
                # next_pos = pos + unit_vec * self.edge_range * 2
                # next_pose_for_map = (next_pos[0], next_pos[2], rot_vec[1])
                # next_pose_on_grid_map_cm = self.local_mapper.get_mapper_pose_from_sim_pose(next_pose_for_map,
                #                                                                            pose_origin_for_map)
                # next_pose_on_grid_map = self.local_mapper.get_map_grid_from_sim_pose_cm(next_pose_on_grid_map_cm)
                # if self.local_mapper.is_traversable(curr_local_map, pose_on_map, next_pose_on_grid_map):
                #     cand_node_info['next_node'] = {'position': next_pos, 'rotation': cand_rot, 'heading_idx': cur_heading_idx}
                # else:
                cand_node_info['next_node'] = None

                cand_nodes.append(cand_node_info)
                free_cand_nodes[angle_bias + i] = 1

        # cand_nodes.append({'position': cand_pos, 'rotation': cand_rot})
        #
        pano_split_images = self.get_dirc_imgs_from_pano(pano_rgb)
        cand_split_images = pano_split_images[np.where(free_cand_nodes == 1)[0]]

        valid_cand_nodes = []
        # similarity, cand_split_feat = self.common_sense_model.clip.get_text_image_sim(text, cand_split_images,
        #                                                                               out_img_feat=True)
        if len(cand_split_images) > 0:
            torch.set_num_threads(1)
            cand_image_feat = self.common_sense_model.clip.get_image_feat(cand_split_images)
            # cm_score, _ = self.common_sense_model.text_image_score(self.goal_place_text_feat, cand_image_feat, feat=True)
            for i in range(len(cand_nodes)):
                # if not self.graph_map.check_node_exist(cand_nodes[i]['position']):
                    # value = np.round(np.max(similarity, axis=1), 3)
                    # cand_nodes[i]['clip_feat'] = cand_split_feat[i]
                    # cand_nodes[i]['value'] = value[i]

                cand_nodes[i]['clip_feat'] = cand_image_feat[i]
                if cand_nodes[i]['next_node'] is not None:
                    cand_nodes[i]['next_node']['clip_feat'] = cand_image_feat[i]
                # cand_nodes[i]['cm_score'] = cm_score[i]
                valid_cand_nodes.append(cand_nodes[i])

        return valid_cand_nodes






    def check_close_goal(self, pos, goal_position, th=1.0):

        if self.dist_euclidean_floor(pos, goal_position) < th:
        # if np.linalg.norm(pos - goal_position) < th:
        # _, geo_dist = self.check_goal_point_validity(pos, goal_position, is_goal_obj=True)
        # if geo_dist < th:
            return True
        else:
            return False

    def check_close_viewpoint(self, pos, view_points, th=0.1):
        close = False
        for view_point in view_points:
            # if np.linalg.norm(pos - view_point['agent_state']['position']) < th:
            if self.dist_euclidean_floor(pos, view_point['agent_state']['position']) < th:
                close = True
                break
        return int(close)

    def check_close_goal_det(self, rgb, depth, vis=False):
        obj_min_dist = 9999
        closest_obj_id = None
        rgb = rgb[:,:,:3]
        if vis:
            img, pred_classes, scores, pred_out, masks, boxes = self.detector.predicted_img(rgb, show=True)
        else:
            pred_classes, scores, pred_out, masks, boxes = self.detector.predicted_img(rgb)
        for i, goal_idx in enumerate(pred_classes):
            if goal_idx == self.goal_class_idx:
                # if np.min(depth[masks[i]]) < th:
                #     close = True
                #     break
                # temp_min_dist = np.min(depth[masks[i]])
                temp_min_dist = np.min(depth[np.nonzero(depth*masks[i])])
                if temp_min_dist < obj_min_dist:
                    obj_min_dist = temp_min_dist
                    closest_obj_id = i

        det_out = {
            'pred_classes': pred_classes,
            'scores': scores,
            'pred_out': pred_out,
            'masks': masks,
            'boxes': boxes,
            'closest_obj_id': closest_obj_id
        }
        if vis:
            det_out['det_img'] = img
        return det_out, obj_min_dist

    def get_position_from_pixel(self, cur_position, cur_rotation, depth, pixel):


        width, height = np.shape(depth)[1], np.shape(depth)[0]
        aspect_ratio = float(width) / float(height)
        fov = np.deg2rad(self.hfov)
        fx = width / 2.0 / np.tan(fov / 2.0)
        fy = fx / aspect_ratio
        cx, cy = width / 2.0, height / 2.0

        z = depth[pixel[0], pixel[1]]
        x = (pixel[0] - cx) * z / fx
        y = (pixel[1] - cy) * z / fy

        rel_position = (-x, y, -z)
        rot = R.from_rotvec(cur_rotation)
        rot.as_matrix()
        target_position = cur_position + rot.apply(rel_position)

        return target_position


    def update_cand_node_to_graph(self, cur_node, cand_nodes):
        if len(cand_nodes) == 0:
            return
        for cand_node_info in cand_nodes:
            cand_node = self.graph_map.add_single_node(cand_node_info['position'])
            # cand_node = self.graph_map.get_node_by_pos(cand_node_info['position'])
            if int(cand_node.nodeid) == len(self.graph_map.nodes)-1: ## new node
                self.graph_map.update_node_goal_category(cand_node, self.goal_class_onehot)
                self.graph_map.update_node_clip_feat(cand_node, cand_node_info['clip_feat'], cand_node_info['heading_idx'])
                self.graph_map.update_node_vis_feat(cand_node)

                torch.set_num_threads(1)
                # cm_score, arg_cm_score = self.common_sense_model.text_image_score(self.goal_place_text_feat, cand_node.vis_feat, feat=True)
                cm_scores, _ = self.common_sense_model.text_image_score(self.goal_place_text_feat,
                                                                        cand_node.vis_feat, feat=True,
                                                                        return_only_max=False)
                # cm_score = np.round(np.max(cm_scores, axis=1),5)
                cm_score = np.round(np.max(np.exp(cm_scores) / np.sum(np.exp(cm_scores)), axis=1), 5)
                arg_cm_score = np.argmax(cm_scores, axis=1)

                cm_info = {
                    'goal_place_name': self.goal_info['category_place'],
                    'goal_place_feat': self.goal_place_text_feat,
                    'goal_place_score': cm_scores,
                }

                self.graph_map.update_node_cm_score(cand_node, cm_score[0] * self.goal_info['category_place_score'][arg_cm_score[0]]
                                                    , cm_name=self.goal_info['category_place'][arg_cm_score[0]]
                                                    , cm_info=cm_info)
                self.graph_map.update_node_feat(cand_node)
                # self.graph_map.update_node_cm_score(cand_node, cand_node_info['cm_score'])

                curr_dist_to_objs, curr_is_valid = self.dist_to_objs(cand_node_info['position'] + self.abs_init_position)
                self.graph_map.update_node_dist_to_objs(cand_node, curr_dist_to_objs)
                self.graph_map.add_edge(cur_node, cand_node)

                if not cand_node_info['next_node'] == None:
                    next_node = self.graph_map.add_single_node(cand_node_info['next_node']['position'])
                    if int(next_node.nodeid) == len(self.graph_map.nodes) - 1:  ## new node
                        self.graph_map.update_node_goal_category(next_node, self.goal_class_onehot)
                        self.graph_map.update_node_clip_feat(next_node, cand_node_info['next_node']['clip_feat'],
                                                             cand_node_info['next_node']['heading_idx'])
                        self.graph_map.update_node_vis_feat(next_node)

                        # cm_score, arg_cm_score = self.common_sense_model.text_image_score(self.goal_place_text_feat, next_node.vis_feat, feat=True)
                        cm_scores, _ = self.common_sense_model.text_image_score(self.goal_place_text_feat,
                                                                                next_node.vis_feat, feat=True,
                                                                                return_only_max=False)
                        # cm_score = np.round(np.max(cm_scores, axis=1),5)
                        cm_score = np.round(np.max(np.exp(cm_scores) / np.sum(np.exp(cm_scores)), axis=1), 5)
                        arg_cm_score = np.argmax(cm_scores, axis=1)

                        cm_info = {
                            'goal_place_name': self.goal_info['category_place'],
                            'goal_place_feat': self.goal_place_text_feat,
                            'goal_place_score': cm_scores,
                        }
                        self.graph_map.update_node_cm_score(next_node,
                                                            cm_score[0] * self.goal_info['category_place_score'][
                                                                arg_cm_score[0]], cm_name=self.goal_info['category_place'][arg_cm_score[0]]
                                                            , cm_info=cm_info)
                        self.graph_map.update_node_feat(next_node)
                        # self.graph_map.update_node_cm_score(cand_node, cand_node_info['cm_score'])

                        curr_dist_to_objs, curr_is_valid = self.dist_to_objs(
                            cand_node_info['next_node']['position'] + self.abs_init_position)
                        self.graph_map.update_node_dist_to_objs(next_node, curr_dist_to_objs)
                        self.graph_map.add_edge(cand_node, next_node)
                    elif np.linalg.norm(np.asarray(cand_node.pos) - np.asarray(next_node.pos)) < self.edge_range * 1.05:
                        self.graph_map.add_edge(cand_node, next_node)

            elif np.linalg.norm(np.asarray(cur_node.pos) - np.asarray(cand_node.pos)) < self.edge_range * 1.05:
                self.graph_map.add_edge(cur_node, cand_node)








    def get_shortest_path(self, start_node_id, end_node_id, adj_mtx):
        start_node_id, end_node_id = int(start_node_id), int(end_node_id)

        graph = csr_matrix(adj_mtx)
        dist_matrix, predecessors = shortest_path(csgraph=graph, directed=False, return_predecessors=True)
        path = [end_node_id]
        while path[-1] != start_node_id:
            path.append(predecessors[start_node_id, path[-1]])
        path.reverse()

        path = [str(i) for i in path]

        return path[1:]



    def do_explicit_action(self, cur_node, action):
        abs_prev_position = self._sim.agents[0].get_state().position
        obs = self._sim.step(action)
        self.abs_position = self._sim.agents[0].get_state().position
        self.path_length += self.dist_euclidean_floor(abs_prev_position, self.abs_position)
        self.abs_rotation = quaternion.as_rotation_vector(self._sim.agents[0].get_state().rotation)
        curr_position = self.abs_position - self.abs_init_position
        curr_rotation = self.abs_rotation


        pano_obs = self.panoramic_obs(obs, semantic=True)
        self.pano_rgb_list.append(pano_obs['rgb_panoramic'])
        self.rgb_list.append(obs['color_sensor'])
        self.depth_list.append(obs['depth_sensor'])

        # det, det_dist = self.check_close_goal_det(obs['color_sensor'], obs['depth_sensor'], vis=True)

        if self.vis_floorplan:
            vis_graph_map = self.vis_topdown_map_with_captions(self.graph_map,
                                                               curr_node=self.cur_node)
                                                               # vis_goal_obj_score=self.goal_class_idx)
            self.vis_info['cur_position'] = curr_position
            self.vis_info['mode'] = 'node search'
            total_frame = self.make_total_frame(obs['color_sensor'], obs['depth_sensor'], vis_graph_map, info=self.vis_info)
            self.vis_traj.append(total_frame)

        self.action_step += 1

        ## update candidate node
        cand_nodes = self.get_cand_node_dirc(self.pano_rgb_list[-1],
                                             self.depth_list[-1], curr_position, curr_rotation)
        self.update_cand_node_to_graph(cur_node, cand_nodes)

        return

    def do_panoramic_action(self, cur_node):
        action = 'turn_left'
        for i in range(int(360/self.act_rot)):
            self.do_explicit_action(cur_node, action)
        return


    def success_evaluation(self, last_position, data):
        # nearest_goal = self.find_nearest_goal(last_position, goal_class_idx)
        # goal_view_points = self.goal_id_to_viewpoints[nearest_goal['id']]
        # success = self.check_close_viewpoint(last_position, goal_view_points, th=0.1)

        # dist = np.linalg.norm(self.env_class_goal_view_point[data['object_category']] - last_position, axis=1)
        dist = self.get_geodesic_distance_to_object_category(last_position, data['object_category'])
        if np.min(dist) < 0.1:
            success = 1.
        else:
            success = 0.
            # dist = np.linalg.norm(self.viewpoint_goal_position - last_position)
            # if np.min(dist) < 0.1:
            #     success = 1.
            # else:
            #     success = 0.



        spl = success * data['info']['geodesic_distance'] / max(self.path_length, data['info']['geodesic_distance'])

        return success, spl



    def do_time_steps(self, data_idx):
        self.abs_position = self._sim.agents[0].get_state().position
        self.abs_rotation = quaternion.as_rotation_vector(self._sim.agents[0].get_state().rotation)
        self.abs_heading = -self.abs_rotation[1] * 180 / np.pi - self.abs_init_heading


        self.abs_heading = self.abs_heading % 360
        if self.abs_heading > 180:
            self.abs_heading -= 180
        if self.abs_heading < 180:
            self.abs_heading += 180

        prev_position = self.abs_position - self.abs_init_position
        prev_rotation = self.abs_rotation #- self.abs_init_rotation
        prev_heading = self.abs_heading - self.abs_init_heading

        max_action_step = False
        last_mile_navi_mode = False
        last_mile_obs = None
        last_mile_det = None

        # try:
        #     path = self.follower.find_path(self.viewpoint_goal_position)
        # except habitat_sim.errors.GreedyFollowerError:



        path = self.follower.find_path(self.goal_position)

        if self.vis_floorplan:
            vis_graph_map = self.vis_topdown_map_with_captions(self.graph_map, curr_node=self.cur_node)


        arrive_node = False

        for frame_idx, action in enumerate(path):
            if action == None:
                break
            step_prev_position = self._sim.agents[0].get_state().position
            obs = self._sim.step(action)
            step_after_position = self._sim.agents[0].get_state().position
            self.path_length += self.dist_euclidean_floor(step_prev_position, step_after_position)
            pano_obs = self.panoramic_obs(obs, semantic=False)
            self.pano_rgb_list.append(pano_obs['rgb_panoramic'])
            self.rgb_list.append(obs['color_sensor'])
            self.depth_list.append(obs['depth_sensor'])

            self.action_step += 1

            if action == 'move_forward':
                arrive_node = False

            curr_state = self._sim.agents[0].get_state()
            curr_position = curr_state.position - self.abs_init_position
            curr_rotation = quaternion.as_rotation_vector(curr_state.rotation) #- self.abs_init_rotation

            if self.action_step > self.max_step:
                max_action_step = True
                break

            if max_action_step:
                break

            if self.vis_floorplan:
                self.vis_info['cur_position'] = curr_position
                self.vis_info['mode'] = 'Exploration'
                total_frame = self.make_total_frame(obs['color_sensor'], obs['depth_sensor'], vis_graph_map,
                                                    info=self.vis_info)
                self.vis_traj.append(total_frame)

            if np.linalg.norm(prev_position-curr_position) > self.edge_range * 0.9:
                arrive_node = True
                # self.cur_node = self.graph_map.add_single_node(curr_position)
                nearest_node_idx, nearest_node_dist = self.graph_map.get_nearest_node(curr_position)
                self.cur_node = self.graph_map.node_by_id[nearest_node_idx]
                # self.graph_map.update_node_pos(self.cur_node, curr_position)
                pano_images = self.get_dirc_imgs_from_pano(pano_obs['rgb_panoramic'])

                torch.set_num_threads(1)
                pano_image_feat = self.common_sense_model.clip.get_image_feat(pano_images)
                cur_heading_idx = int(np.round(-curr_rotation[1] * 180 / np.pi / self.cand_rot_angle)) % self.rot_num
                for i in range(len(pano_images)):
                    dirc_head_idx = (cur_heading_idx -4 + i) % self.rot_num
                    self.graph_map.update_node_clip_feat(self.cur_node, pano_image_feat[i], dirc_head_idx)
                self.graph_map.update_node_vis_feat(self.cur_node)

                # cm_score, arg_cm_score = self.common_sense_model.text_image_score(self.goal_place_text_feat, self.cur_node.vis_feat, feat=True)
                cm_scores, _ = self.common_sense_model.text_image_score(self.goal_place_text_feat,
                                                                        self.cur_node.vis_feat, feat=True,
                                                                        return_only_max=False)
                # cm_score = np.round(np.max(cm_scores, axis=1),5)
                cm_score = np.round(np.max(np.exp(cm_scores) / np.sum(np.exp(cm_scores)), axis=1), 5)
                arg_cm_score = np.argmax(cm_scores, axis=1)

                cm_info = {
                    'goal_place_name': self.goal_info['category_place'],
                    'goal_place_feat': self.goal_place_text_feat,
                    'goal_place_score': cm_scores,
                }
                self.graph_map.update_node_cm_score(self.cur_node, cm_score[0] * self.goal_info['category_place_score'][arg_cm_score[0]]
                                                    , cm_name=self.goal_info['category_place'][arg_cm_score[0]]
                                                    , cm_info=cm_info)
                self.graph_map.update_node_visited(self.cur_node)
                self.graph_map.update_node_feat(self.cur_node)

                curr_dist_to_objs, curr_is_valid = self.dist_to_objs(curr_state.position)
                self.graph_map.update_node_dist_to_objs(self.cur_node, curr_dist_to_objs)

                prev_position = curr_position
                prev_rotation = curr_rotation

            if arrive_node:
                cand_nodes = self.get_cand_node_dirc(self.pano_rgb_list[-1], self.depth_list[-1], curr_position, curr_rotation)
                self.update_cand_node_to_graph(self.cur_node, cand_nodes)

                if self.vis_floorplan:
                    vis_graph_map = self.vis_topdown_map_with_captions(self.graph_map, curr_node=self.cur_node)

            # view_point_dist = np.linalg.norm(self.env_class_goal_view_point[self.goal_obj_names[self.goal_class_idx]] - curr_state.position, axis=1)
            # if np.min(view_point_dist) < 0.1:
            #     break


        return

    def get_data(self, env_idx, tot_env_num):


        self.init_random()
        self.env_name = self._sim_settings["scene"].split('/')[-1].split('.')[0]
        self.obj_list = self._sim.semantic_scene.objects
        self.obj_list = [obj for obj in self.obj_list if not obj == None]


        valid_obj_list = []
        for obj in self.obj_list:
            if obj.category.name() in self.obj_names: # and obj.region.level.id == '0':
                valid_obj_list.append(obj)

        num_obj = len(valid_obj_list)


        self.env_obj_info = []
        self.env_goal_obj_info = []
        for i, obj in enumerate(valid_obj_list):
            obj_position = obj.aabb.center
            obj_region_level = obj.region.level.id
            obj_category = self.obj_names.index(obj.category.name())
            self.env_obj_info.append({'position': obj_position, 'sizes':obj.aabb.sizes, 'category':obj_category,
                                      'id':obj.semantic_id,
                                      'level':obj_region_level,
                                      })



        for i, obj in enumerate(self.obj_list):
            if obj.category.name() in self.goal_obj_names:
                obj_position = obj.aabb.center
                obj_region_level = obj.region.level.id
                goal_obj_category = self.goal_obj_names.index(obj.category.name())
                self.env_goal_obj_info.append({'position': obj_position, 'sizes':obj.aabb.sizes,
                                               'category':goal_obj_category, 'id':obj.semantic_id,
                                               'level':obj_region_level
                                               })

        if self.args.dataset == 'mp3d' or self.args.dataset == 'hm3d':
            self.goal_id_to_viewpoints = {}
            for i, obj in enumerate(self.env_goal_obj_info):
                obj_name = self.goal_obj_names[obj['category']]
                if f'{self.env_name}.glb_{obj_name}' in self.dataset['goals_by_category']:
                    for o in self.dataset['goals_by_category'][f'{self.env_name}.glb_{obj_name}']:
                        if o['object_id'] == obj['id']:
                            self.goal_id_to_viewpoints[obj['id']] = o['view_points']
                            break

            self.env_class_goal_obj = {}
            # self.env_class_goal_obj_point = {}
            self.env_class_goal_view_point = {}
            for obj_name in self.goal_obj_names:
                self.env_class_goal_obj[obj_name] = []
                # self.env_class_goal_obj_point[obj_name] = []
                self.env_class_goal_view_point[obj_name] = []

            for obj in self.obj_list:
                if obj.category.name() in self.goal_obj_names:
                    self.env_class_goal_obj[obj.category.name()].append(obj)
                    # self.env_class_goal_obj_point[obj.category.name()].append(obj.aabb.center)
                    if obj.semantic_id in self.goal_id_to_viewpoints:
                        view_point_positions = [v['agent_state']['position'] for v in self.goal_id_to_viewpoints[obj.semantic_id]]
                        self.env_class_goal_view_point[obj.category.name()].extend(view_point_positions)

            self.env_class_goal_view_point = {k:np.array(v) for k,v in self.env_class_goal_view_point.items()}



        ## set floor 0 vertical range
        if self.args.dataset == 'mp3d':
            self.set_level_range()

        # self.room_names = set([])
        # self.room_list = self._sim.semantic_scene.regions
        # self.room_info = []
        # self.avail_room_info = []
        # for room in self.room_list:
        #     if room.level.id == '0':
        #         self.room_names.add(room.category.name())
        #         self.room_info.append(
        #             {'center': room.aabb.center, 'sizes': room.aabb.sizes, 'category': room.category.name()})

        valid_traj_list = []

        goal_traj_nums = np.zeros(len(self.goal_obj_names))
        for epi in self.dataset['episodes']:
            # if goal['info']['geodesic_distance'] < 1.0:
            #     continue
            if self.args.dataset == 'mp3d' or self.args.dataset == 'hm3d':
                if abs(epi['start_position'][1] - epi['info']['best_viewpoint_position'][1]) > 1.0:
                    continue
            if epi['object_category'] in self.goal_obj_names:
                epi['goals'] = self.dataset['goals_by_category']['{}.glb_{}'.format(self.env_name, epi['object_category'])]
                valid_traj_list.append(epi)
                goal_traj_nums[self.goal_obj_names.index(epi['object_category'])] += 1
                goal_category = epi['object_category']


        print("Get env information done")

        self.goal_category_room, self.goal_category_room_feat, self.goal_category_room_score = \
                                        self.init_commonsense_candidate_room(self.goal_obj_names, mp3d_room_names)
        self.goal_category_feat = self.common_sense_model.clip.get_text_feat(gibson_goal_obj_names).type(torch.float32).cpu()


        if len(self.env_goal_obj_info) == 0:
            self._sim.close()
            del self._sim
            return


        self.env_size = abs(self._sim.semantic_scene.aabb.sizes[0] * self._sim.semantic_scene.aabb.sizes[0])
        # self.n_for_env = int(self.env_size/4)

        self.pathfinder = self._sim.pathfinder
        self.follower = habitat_sim.GreedyGeodesicFollower(self.pathfinder, self._sim.agents[0], goal_radius=0.1)

        lower_bound, upper_bound = self.pathfinder.get_bounds()

        env_start_time = time.time()
        src_start_time = time.time()

        data_idx = 0
        # for idx in range(self.n_for_env):

        success_results = {
            'total': {'success': 0, 'spl': 0, 'count': 0},
            'easy': {'success': 0, 'spl': 0, 'count': 0},
            'medium': {'success': 0, 'spl': 0, 'count': 0},
            'hard': {'success': 0, 'spl': 0, 'count': 0},
        }
        total_success, total_spl, easy_success, easy_spl, medium_success, medium_spl, hard_success, hard_spl = \
            [], [], [], [], [], [], [], []

        obj_success_results, obj_success_list, obj_spl_list = {}, {}, {}
        for obj_name in self.goal_obj_names:
            obj_success_results[obj_name] = {'success': 0, 'spl': 0, 'count': 0}
            obj_success_list[obj_name], obj_spl_list[obj_name] = [], []





        max_data_num = self.args.n_for_env
        if self.data_type == 'val':
            max_data_num = max_data_num // 10



        range_trial = 0

        while True:

            if data_idx > max_data_num:
                break

            data_dir = f"{self.args.save_dir}/{self.data_type}/{self.env_name}/{self.env_name}_{data_idx:04d}"
            if os.path.exists(data_dir):
                if os.path.exists(os.path.join(data_dir, f"{self.env_name}_{data_idx:04d}.npy")):
                    print(f"File Exit {self.env_name}_{data_idx:04d}")
                    exist_data = np.load(os.path.join(data_dir, f"{self.env_name}_{data_idx:04d}.npy"), allow_pickle=True).item()
                    if len(exist_data['action']) <= 500:
                        data_idx += 1
                        continue
                    else:
                        print(f"Invalid data {self.env_name}_{data_idx:04d} : too long trajectory > 500")

                os.system(f"rm -r {data_dir}")


            # difficulty = random.randint(0, 2)




            if data_idx < max_data_num/3:
                min_dist, max_dist = 0.5, 5.0
            elif data_idx < max_data_num * 2 / 3:
                min_dist, max_dist = 5.0, 10.0
            else:
                min_dist, max_dist = 10.0, None


            start_state, self.goal_position, no_valid, ged_dist = self.random_inital_traj_agent(min_dist=min_dist, max_dist=max_dist)
            if no_valid:
                continue



            while True:
                self.goal_class_idx = random.randint(0, len(self.goal_obj_names)-1)
                if len(self.env_class_goal_view_point[self.goal_obj_names[self.goal_class_idx]]) > 0:
                    break

            ## -- floor plan -- ##
            if self.vis_floorplan:
                self.calculate_navmesh()
                self.update_cur_floor_map()

            self.cur_data_idx = data_idx
            self.action_step = 0
            self.path_length = 1e-5

            agent_id = self._sim_settings["default_agent"]
            agent = self._sim.initialize_agent(agent_id)
            self.local_navi_module.reset_sim_and_agent(self._sim, agent)

            agent.set_state(start_state)



            self.abs_init_position = start_state.position
            self.abs_init_rotation = quaternion.as_rotation_vector(start_state.rotation)
            self.abs_init_heading = -self.abs_init_rotation[1] * 180 / np.pi

            self.abs_position = start_state.position
            self.abs_rotation = quaternion.as_rotation_vector(start_state.rotation)
            self.abs_heading = -self.abs_rotation[1] * 180 / np.pi

            self.cur_position = np.zeros([3])
            self.cur_rotation = quaternion.as_rotation_vector(start_state.rotation)
            self.cur_heading = np.zeros([3])

            ## -- floor plan -- ##
            if self.vis_floorplan:
                self.cur_map = self.map.copy()

            ## init graph map
            self.graph_map = GraphMap(self.args)
            self.graph_map.goal_text_clip_feat = self.goal_category_feat[self.goal_class_idx]
            obs = self._sim.get_sensor_observations()

            # if self.args.dataset == 'mp3d' or self.args.dataset == 'hm3d':
            #     goal_info = self._sim.semantic_scene.objects[traj['info']['closest_goal_object_id']]
            #     self.goal_info = self.update_goal_info(goal_info)
            #     self.goal_info['goal_name'] = goal_info.category.name()
            #     self.goal_class_idx = self.goal_obj_names.index(goal_info.category.name())

            # elif self.args.dataset == 'gibson':
            #     self.goal_info = {}
            #     self.goal_info['category'] = traj['object_category']
            #     # self.goal_info['goal_name'] = traj['object_category']
            #     self.goal_class_idx = self.goal_obj_names.index(traj['object_category'])

            self.goal_class_onehot = torch.zeros([len(self.goal_obj_names)])
            self.goal_class_onehot[self.goal_class_idx] = 1

            self.goal_info = {'category': self.goal_obj_names[self.goal_class_idx]}

            self.goal_info['category_place'] = self.goal_category_room[self.goal_info['category']]
            self.goal_info['category_place_score'] = self.goal_category_room_score[self.goal_info['category']]
            self.goal_place_text_feat = self.goal_category_room_feat[self.goal_info['category']]

            # dist_to_objs, is_valid_point = self.dist_to_objs(start_state.position)

            if self.vis_floorplan:
                self.vis_traj = []
                self.vis_info = {
                    'target_goal': self.goal_obj_names[self.goal_class_idx],
                    'mode': 'Exploration',
                    'cur_position': self.cur_position,
                    'obj_position': None,
                }


            pano_obs = self.panoramic_obs(obs, semantic=True)
            self.pano_rgb_list = [pano_obs['rgb_panoramic']]
            self.rgb_list = [obs['color_sensor']]
            self.depth_list = [obs['depth_sensor']]
            # det, det_dist = self.check_close_goal_det(obs['color_sensor'], obs['depth_sensor'], vis=True)


            # set initial node
            self.cur_node = self.graph_map.add_single_node(self.cur_position)
            # cur_node = self.graph_map.get_node_by_pos(self.cur_position)

            # similarity, pano_clip_feat = self.common_sense_model.clip.get_text_image_sim(self.goal_info['category'], pano_images, out_img_feat=True)
            # self.graph_map.update_node_clip_feat(self.cur_node, pano_clip_feat[0])
            # value = np.round(np.max(similarity, axis=1), 3)

            self.graph_map.update_node_goal_category(self.cur_node, self.goal_class_onehot)
            pano_images = self.get_dirc_imgs_from_pano(self.pano_rgb_list[-1])

            torch.set_num_threads(1)
            pano_images_feat = self.common_sense_model.clip.get_image_feat(pano_images).type(torch.float32).detach().cpu()
            cur_heading_idx = int(np.round(-self.cur_rotation[1] * 180 / np.pi / self.cand_rot_angle)) % self.rot_num
            for i in range(len(pano_images)):
                dirc_head_idx = (cur_heading_idx - 4 + i) % self.rot_num
                self.graph_map.update_node_clip_feat(self.cur_node, pano_images_feat[i], dirc_head_idx)
            self.graph_map.update_node_vis_feat(self.cur_node)

            # cm_score, arg_cm_score = self.common_sense_model.text_image_score(self.goal_place_text_feat, self.cur_node.vis_feat, feat=True)
            cm_scores, _ = self.common_sense_model.text_image_score(self.goal_place_text_feat, self.cur_node.vis_feat, feat=True, return_only_max=False)
            # cm_score = np.round(np.max(cm_scores, axis=1),5)
            cm_score = np.round(np.max(np.exp(cm_scores) / np.sum(np.exp(cm_scores)), axis=1),5)
            arg_cm_score = np.argmax(cm_scores, axis=1)

            cm_info={
                'goal_place_name': self.goal_info['category_place'],
                'goal_place_feat': self.goal_place_text_feat,
                'goal_place_score': cm_scores,
            }
            self.graph_map.update_node_cm_score(self.cur_node, cm_score[0] * self.goal_info['category_place_score'][arg_cm_score[0]]
                                                , cm_name=self.goal_info['category_place'][arg_cm_score[0]]
                                                , cm_info=cm_info)
            curr_dist_to_objs, curr_is_valid_point = self.dist_to_objs(self.abs_position)
            if not curr_is_valid_point:
                continue
            self.graph_map.update_node_dist_to_objs(self.cur_node, curr_dist_to_objs)
            self.graph_map.update_node_visited(self.cur_node)
            self.graph_map.update_node_feat(self.cur_node)
            self.graph_map.update_node_is_start(self.cur_node)


            ## update candidate node

            # cand_nodes = self.get_cand_node(self.pano_rgb_list[-1], self.cur_position, self.cur_heading, self.goal_info)
            cand_nodes = self.get_cand_node_dirc(self.pano_rgb_list[-1], self.depth_list[-1], self.cur_position, self.abs_init_rotation)
            self.update_cand_node_to_graph(self.cur_node, cand_nodes)

            if self.vis_floorplan:
                vis_graph_map = self.vis_topdown_map_with_captions(self.graph_map,
                                                                   curr_node=self.cur_node,
                                                                   vis_goal_obj_score=self.goal_class_idx)
                self.vis_info['cur_position'] = self.cur_position
                self.vis_info['mode'] = 'Exploration'
                total_frame = self.make_total_frame(obs['color_sensor'], obs['depth_sensor'], vis_graph_map, info=self.vis_info)
                self.vis_traj.append(total_frame)


            data_dir = f"{self.args.save_dir}/{self.data_type}/{self.env_name}"
            if not os.path.exists(data_dir): os.makedirs(data_dir)


            try:
                # self.viewpoint_goal_position = traj['info']['best_viewpoint_position']

                self.do_panoramic_action(self.cur_node)
                self.do_time_steps(data_idx)

            except KeyboardInterrupt:
                print("KeyboardInterrupt")
                break

            except:
                pass

            if not os.path.exists(f'{self.args.save_dir}/{self.data_type}/{self.env_name}/{self.env_name}_{data_idx:04d}'):
                os.makedirs(f'{self.args.save_dir}/{self.data_type}/{self.env_name}/{self.env_name}_{data_idx:04d}')

            self.save_rgbd_video(self.rgb_list, self.depth_list, self.args.save_dir, self.env_name, data_idx)
            # self.save_semantic_video(self.semantic_list, self.args.save_dir, self.env_name, data_idx)
            #
            # self.save_obj_data(obj_data, self.args.save_dir, self.env_name, data_idx, tot=True)

            with open(f'{self.args.save_dir}/{self.data_type}/{self.env_name}/{self.env_name}_{data_idx:04d}/graph.pkl',
                      'wb') as f:
                pickle.dump(self.graph_map, f)

            cur_goal_obj_category_name = self.goal_info['category']
            if self.vis_floorplan:
                vis_save_dir = f'{self.args.save_dir}/{self.data_type}/{self.env_name}/{self.env_name}_{data_idx:04d}/vis_{cur_goal_obj_category_name}.png'
                self.vis_topdown_map_with_captions(self.graph_map, save_dir=vis_save_dir,
                                                   curr_node=self.cur_node)
                self.save_video(self.vis_traj, self.args.save_dir, self.env_name, data_idx)

            data_idx += 1
            print(f"[{data_idx}/{max_data_num}] {self.env_name}  Done,   Time : {time.time()-env_start_time}")

        # data_dir = f"{self.args.save_dir}/{self.data_type}/{self.env_name}"
        # if not os.path.exists(data_dir):
        #     err_data_dir = f"{self.args.save_dir}/err/{self.env_name}"
        #     os.makedirs(err_data_dir)


        self._sim.close()
        del self._sim



        return

