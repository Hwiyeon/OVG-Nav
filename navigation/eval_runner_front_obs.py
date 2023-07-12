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
import torch.nn as nn
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
from utils.obj_category_info import assign_room_category, obj_names_det as obj_names, gibson_goal_obj_names, mp3d_goal_obj_names, room_names, mp3d_room_names, rednet_obj_names

from tqdm import tqdm
import pickle

from modules.detector.detector_mask import Detector
from modules.detector.rednet_semantic_prediction import SemanticPredRedNet
from modules.free_space_model.inference import FreeSpaceModel
from modules.comet_relation.inference import CommonSenseModel
from modules.visual_odometry.keypoint_matching import KeypointMatching
from goal_dist_pred.model_value_graph_0607 import TopoGCN_v2_pano_goalscore as ValueModel

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

        if self.args.goal_cat == 'mp3d_21':
            self.goal_obj_names = rednet_obj_names

        self.pix_num = args.width*args.height
        self.cand_angle = np.arange(-120, 240, args.cand_rot)
        self.cand_angle_bias = list(self.cand_angle).index(0) # 0 degree is the center
        self.edge_range = args.edge_range
        self.last_mile_range = args.last_mile_range
        self.goal_det_dist = args.success_dist - args.move_forward
        self.goal_obs_consistency_th = 1  # number of time steps that the goal is visible for the goal to be considered as correctly detected

        self.vo_height = args.front_height
        self.vo_width = args.front_width
        self.vo_hfov = args.front_hfov

        self.height = args.height
        self.width = args.width

        self.pano_width = args.pano_width
        self.pano_height = args.pano_height

        self.hfov = args.hfov
        self.camera_height = args.sensor_height

        self.step_size = args.move_forward
        self.follower_goal_radius = 0.75 * self.step_size
        self.act_rot = args.act_rot
        self.cand_rot_angle = args.cand_rot
        self.rot_num = len(self.cand_angle)
        self.max_local_action_trial = 50
        self.max_step = args.max_step

        # self.vo_pred_model = VO_prediction(args.vo_config)
        self.depth_scale = np.iinfo(np.uint16).max

        self.goal_cat = args.goal_cat
        if args.goal_cat == 'mp3d':
            self.detector = Detector(args, self.det_COI)
        elif args.goal_cat == 'mp3d_21':
            self.detector = SemanticPredRedNet(args)
        self.free_space_model = FreeSpaceModel(args)
        self.common_sense_model = CommonSenseModel(args)
        self.noisy_pose = args.noisy_pose
        self.vo_model = KeypointMatching(args)

        self.value_model = ValueModel(self.args)
        # self.value_model = nn.DataParallel(self.value_model).cuda()
        # self.value_model.load_state_dict(torch.load(self.args.value_model))
        state_dict = torch.load(self.args.value_model)
        new_state_dict = {}
        for k, v in state_dict.items():
            if 'module' in k:
                k = k.replace('module.', '')
            new_state_dict[k] = v
        self.value_model.load_state_dict(new_state_dict)
        self.value_model = self.value_model.to(f'cuda:{args.model_gpu}')
        self.value_model.eval()

        self.local_navi_module = LocalNavigation(self.args, self.vo_model)
        self.local_agent = LocalAgent(self.args)
        self.local_mapper = build_mapper(self.args)



        self.vis_floorplan = args.vis_floorplan
        self.use_oracle = args.use_oracle
        self.cm_type = args.cm_type  ### 'comet or mp3d'



    def save_rgbd_video(self, rgb_list, depth_list, save_dir, panoramic=False):

        # data_dir = f"{save_dir}/{self.data_type}/{env_name}/{env_name}_{idx:04d}"
        # if not os.path.exists(data_dir): os.makedirs(data_dir)

        if panoramic:
            width = self.pano_width
            height = self.pano_height
            rgb_name = 'pano_rgb'
            depth_name = 'pano_depth'
        else:
            width = 320  # self.vo_width
            height = 240   # self.vo_height
            rgb_name = 'rgb'
            depth_name = 'depth'

        video = cv2.VideoWriter(f'{save_dir}/{rgb_name}.avi', cv2.VideoWriter_fourcc(*'XVID'), 5,
                                (width, height))
        for image in rgb_list:
            image = cv2.cvtColor((image[:, :, :3] / 255.).astype(np.float32), cv2.COLOR_RGB2BGR)
            image = cv2.resize(image, (width, height))
            video.write((image * 255).astype(np.uint8))
        video.release()

        video = cv2.VideoWriter(f'{save_dir}/{depth_name}.avi', cv2.VideoWriter_fourcc(*'XVID'), 5,
                                (width, height), isColor=False)
        for depth_obs in depth_list:
            # norm_depth = np.where(depth_obs < 10, depth_obs/10., 1.).astype(np.float32)
            # norm_depth = (norm_depth * np.iinfo(np.uint16).max).astype(np.uint16)
            depth_obs = cv2.resize(depth_obs, (width, height))
            depth_obs = (np.clip(depth_obs, 0.1, 10.) / 10.).astype(np.float32)
            depth_obs = (depth_obs * self.depth_scale).astype(np.uint16) / self.depth_scale
            depth_obs = (depth_obs * 255).astype(np.uint8)
            video.write(depth_obs)
        video.release()

    def save_video(self, frame_list, save_dir):
        # data_dir = f"{save_dir}/{self.data_type}/{env_name}/{env_name}_{idx:04d}"
        if not os.path.exists(save_dir): os.makedirs(save_dir)

        width = np.shape(frame_list[0])[1]
        height = np.shape(frame_list[0])[0]

        video = cv2.VideoWriter(f'{save_dir}/graph.avi', cv2.VideoWriter_fourcc(*'XVID'), 5,
                                (width, height))
        for image in frame_list:
            image = cv2.cvtColor((image[:, :, :3] / 255.).astype(np.float32), cv2.COLOR_RGB2BGR)
            video.write((image * 255).astype(np.uint8))
        video.release()

    def make_total_frame(self, rgb, depth, graph, local_map, pano_rgb, info):
        rh, rw = np.shape(rgb)[:2]
        rh, rw = int(rh/2), int(rw/2)
        small_rgb = cv2.resize(rgb, (rw, rh))
        small_depth = cv2.resize(depth, (rw, rh))
        small_depth = ((np.clip(small_depth, 0.1, 10.) / 10.) * 255).astype(np.uint8)
        gh, gw = np.shape(graph)[:2]
        gh, gw = rh*2, int(rh*2 * gw / gh)
        ph, pw = np.shape(pano_rgb)[:2]

        lh, lw = np.shape(local_map)[:2]
        local_map = cv2.flip(local_map, 1)
        # lh, lw = int(lh/2), int(lw/2)
        # local_map = cv2.resize(local_map, (lw, lh))

        small_graph = cv2.resize(graph, (gw, gh))
        max_h = max(rh*2, gh, lh)
        max_w = max(rw+gw+lw, pw)

        frame = np.zeros([max_h+ph, max_w, 3])
        frame[:rh, :rw, :] = small_rgb[:, :, :3]
        frame[rh:rh*2, :rw, :] = np.tile(small_depth[:, :, np.newaxis], [1, 1, 3])
        frame[:gh, rw:rw+gw, ] = small_graph
        frame[:lh, rw+gw:rw+gw+lw, ] = local_map[:, :, :3]
        frame[max_h:, :pw, ] = pano_rgb[:, :, :3]
        frame = frame.astype(np.uint8)


        ## text
        text1 = "Target object goal: {}   Mode: {}".format(info['target_goal'], info['mode'])
        text2 = "Position: {}".format(info['cur_position'])
        font_color = (255, 255, 255)
        text_size, _ = cv2.getTextSize(text1, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        # text_position1 = (int((frame.shape[1] - text_size[0]) / 2), frame.shape[0] + text_size[1] * 2 + 10)
        # text_position2 = (int((frame.shape[1] - text_size[0]) / 2), frame.shape[0] + text_size[1] * 2 + 25)
        text_position1 = (10, frame.shape[0] + text_size[1] * 2 + 10)
        text_position2 = (10, frame.shape[0] + text_size[1] * 2 + 25)
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

    def init_commonsense_candidate_room(self, goal_names, candidate_names):
        goal_category_room = {}
        goal_category_room_feat = {}
        goal_category_room_score = {}

        cand_category_room = {}
        cand_category_room_feat = {}
        cand_category_room_score = {}
        cand_room_feat = self.common_sense_model.clip.get_text_feat(candidate_names).type(torch.float32)
        for i, goal_name in enumerate(goal_names):
            pred_words = self.common_sense_model.gen_pred_words(self.goal_obj_names[i] + ' in an indoor space',
                                                                num_generate=10)
            # pred_words = pred_words[0]
            pred_words_feat = self.common_sense_model.clip.get_text_feat(pred_words).type(torch.float32)

            goal_category_room[goal_name] = pred_words
            goal_category_room_feat[goal_name] = pred_words_feat.cpu()
            goal_category_room_score[goal_name] = np.ones_like(pred_words).astype(float)

            value, indice = torch.max(
                self.common_sense_model.clip.get_sim_from_feats(cand_room_feat, pred_words_feat, normalize=True).type(
                    torch.float32),
                dim=0)
            topk = torch.topk(value, k=len(candidate_names), largest=True).indices.detach().cpu().numpy()
            cand_category_room[goal_name] = [candidate_names[i] for i in topk]
            cand_category_room_feat[goal_name] = torch.stack([cand_room_feat[i].detach().cpu() for i in topk])
            cand_category_room_score[goal_name] = np.array([value[i].detach().cpu().numpy() for i in topk])

        return goal_category_room, goal_category_room_feat, goal_category_room_score, cand_category_room, cand_category_room_feat, cand_category_room_score


    def calculate_navmesh(self):
        navmesh_settings = habitat_sim.NavMeshSettings()
        navmesh_settings.set_defaults()
        navmesh_success = self._sim.recompute_navmesh(self._sim.pathfinder, navmesh_settings, include_static_objects=True)
        print("navmesh_success ", navmesh_success )
        num_level = len(self.level_range)
        self.level_navmesh = []
        for i in range(num_level):
            tdv = TopdownView(self._sim, self.args.dataset, data_dir=self.args.floorplan_data_dir)
            # scene_height = np.mean(self.level_range[i])
            # scene_height = self.level_range[i][0] + self.camera_height
            scene_height = np.mean([goal['position'][1] for goal in self.env_goal_obj_info if goal['level'] == str(i)])
            tdv.draw_top_down_map(height=scene_height, floor=i)
            self.level_navmesh.append({
                'tdv': tdv,
                'scene_height': scene_height
            })

        # self.tdv = TopdownView(self._sim, self.args.dataset, data_dir=self.args.floorplan_data_dir)
        # self.scene_height = self._sim.agents[0].state.position[1]
        # self.tdv.draw_top_down_map(height=self.scene_height)



    def get_topdown_floorplan(self, tdv, scene_height):  ## not using
        # if abs(self._sim.agents[0].state.position[1] - self.scene_height) > 0.5:
        #     self.scene_height = self._sim.agents[0].state.position[1]
        #     self.tdv.draw_top_down_map(height=self.scene_height)
        tdv.draw_top_down_map(height=scene_height)
        return tdv.rgb_top_down_map

    def update_cur_floor_map(self):
        # self.map = get_topdown_map_from_sim(self._sim, meters_per_pixel=0.02)
        # self.map_size = np.shape(self.map)
        # self.map = self.get_topdown_floorplan()
        # self.map = cv2.cvtColor(self.map, cv2.COLOR_BGR2RGB)
        # self.map_size = np.shape(self.map)

        self.map, self.map_size = [], []
        for i in range(len(self.level_range)):
            tdv = self.level_navmesh[i]['tdv']
            # scene_height = self.level_navmesh[i]['scene_height']
            # tdv.draw_top_down_map(height=scene_height)
            map = tdv.rgb_top_down_map
            map = cv2.cvtColor(map, cv2.COLOR_BGR2RGB)
            self.map.append(map)
            map_size = np.shape(map)
            self.map_size.append(map_size)

        print(f'Visualize floorplan done with level {len(self.level_range)}')



    def get_vis_grid_pose(self, pose, lv):
        if len(pose) == 3:
            pose = np.array(pose)
            grid_y, grid_x = to_grid(pose[2], pose[0], self.map_size[lv], self._sim)
        elif len(pose) == 2:
            grid_y, grid_x = to_grid(pose[1], pose[0], self.map_size[lv], self._sim)
        return (grid_x, grid_y)

    def node_value_by_obj_dist(self, dist, max_dist=15.0):
        return max(1 - dist / max_dist, 0)

    def vis_obj_viewpoint_on_floormap(self):
        self.goal_map = []

        for lv in range(len(self.level_range)):
            level_map = {}
            for goal_idx, obj_name in enumerate(self.goal_obj_names):
                level_map[obj_name] = np.copy(self.map[lv])

                # ## --- draw goal object bboxes --- ##
                # shapes = np.zeros_like(self.map[lv], np.uint8)
                # for obj in self.env_goal_obj_info:
                #     if obj['category'] == goal_idx and int(obj['level']) == lv:
                #         grid_bbox = self.get_bbox_from_pos_size(obj['position'], obj['sizes'], lv)
                #         grid_bbox = grid_bbox.tolist()
                #
                #         cv2.rectangle(shapes, tuple(grid_bbox[0]), tuple(grid_bbox[1]), (255, 128, 0), -1)
                # alpha = 0.3
                # mask = np.repeat(np.sum(shapes, axis=2).astype(bool)[:, :, np.newaxis], 3, axis=2)
                # level_map[obj_name][mask] = cv2.addWeighted(level_map[obj_name], alpha, shapes, 1 - alpha, 0)[mask]

                # ## --- draw viewpoints --- ##
                # shapes = np.zeros_like(self.map[lv], np.uint8)
                # for pos in self.env_class_goal_view_point_level[obj_name][str(lv)]:
                #     ## view points in dataset ##
                #     node_grid = self.get_vis_grid_pose(pos, lv)
                #     shapes = cv2.circle(shapes, node_grid, 3, (255, 255, 0), -1)
                # alpha = 0.7
                # mask = np.repeat(np.sum(shapes, axis=2).astype(bool)[:,:,np.newaxis], 3, axis=2)
                # level_map[obj_name][mask] = cv2.addWeighted(level_map[obj_name], alpha, shapes, 1 - alpha, 0)[mask]

                ## --- draw goal object centers --- ##
                for pos in np.array([obj['position'] for obj in self.env_goal_obj_info if (obj['category'] == goal_idx and int(obj['level']) == lv)]):
                    ## goal object positions in dataset
                    node_grid = self.get_vis_grid_pose(pos, lv)
                    level_map[obj_name] = cv2.circle(level_map[obj_name], node_grid, 10, (255, 200, 0), -1)

            self.goal_map.append(level_map)


        print(f'Visualize goal object viewpoint on floorplan done with level {len(self.level_range)} and goal obj class {len(self.goal_obj_names)}')


    def vis_topdown_graph_map(self, vis_map, graph_map, vis_obj_score=None, curr_node_id=None, curr_goal_node_id=None,
                              bias_position=None, curr_goal_position=None, visited_positions=None):
        node_list = list(graph_map.node_by_id.values())

        for edge in list(graph_map.edges):
            if not edge.draw:
                pos1 = np.array(edge.nodes[0].pos) if edge.nodes[0].vis_pos is None else np.array(edge.nodes[0].vis_pos)
                pos2 = np.array(edge.nodes[1].pos) if edge.nodes[1].vis_pos is None else np.array(edge.nodes[1].vis_pos)
                node_grid1 = self.get_vis_grid_pose(pos1 + bias_position, self.curr_level)
                node_grid2 = self.get_vis_grid_pose(pos2 + bias_position, self.curr_level)
                # node_grid1 = self.get_vis_grid_pose(np.array(edge.nodes[0].pos) + bias_position, self.curr_level)
                # node_grid2 = self.get_vis_grid_pose(np.array(edge.nodes[1].pos) + bias_position, self.curr_level)
                vis_map = cv2.line(vis_map, node_grid1, node_grid2, (0, 64, 64), 5)
                edge.draw = True

        # cm_scores = []
        # for node in node_list:
        #     cm_scores.append(node.cm_score)
        # cm_scores = np.array(cm_scores)
        # cm_scores = np.exp(cm_scores) / np.sum(np.exp(cm_scores))

        for idx, node in enumerate(node_list):

            # if node.draw and node.nodeid != curr_node_id and node.nodeid != curr_goal_node_id:
            #     continue

            node_pos = np.array(node.pos) if node.vis_pos is None else np.array(node.vis_pos)
            node_grid = self.get_vis_grid_pose(node_pos + bias_position, self.curr_level)
            # node_grid = self.get_vis_grid_pose(np.array(node.pos) + bias_position, self.curr_level)
            if vis_obj_score is not None:
                # color = (np.array((0, 255, 0)) * self.node_value_by_obj_dist(node.dist_to_objs[vis_obj_score])).astype(int)
                # color = (np.array((0, 255, 0)) * node.cm_score).astype(int)
                # color = tuple([color[i].item() for i in range(3)])
                color = (0, 255, 0)

                # cand_color = (np.array((0, 0, 255)) * self.node_value_by_obj_dist(
                #     node.dist_to_objs[vis_obj_score])).astype(int)
                # cand_color = (np.array((0, 0, 255)) * node.cm_score).astype(int)
                cand_color = (np.array((0, 0, 255))).astype(int)
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
                # vis_map = cv2.rectangle(vis_map, (node_grid[0] - 8, node_grid[1] - 8),
                #                         (node_grid[0] + 8, node_grid[1] + 8),
                #                         goal_color, -1)
            else:
                vis_map = cv2.circle(vis_map, node_grid, 10, cand_color, -1)

            node.draw = True

        if visited_positions is not None:
            for pos in visited_positions:
                node_grid = self.get_vis_grid_pose(pos + bias_position, self.curr_level)
                vis_map = cv2.circle(vis_map, node_grid, 5, (125, 0, 0), -1)

        if curr_goal_position is not None:
            node_grid = self.get_vis_grid_pose(curr_goal_position + bias_position, self.curr_level)
            vis_map = cv2.rectangle(vis_map, (node_grid[0] - 8, node_grid[1] - 8), (node_grid[0] + 8, node_grid[1] + 8),
                                    (255, 255, 0), -1)





        return vis_map

    def vis_pos_on_topdown_map(self, pos, lv, vis_map=None, color=(255, 0, 0)):
        if vis_map is None:
            vis_map = self.map[self.curr_level].copy()
        else:
            vis_map = vis_map.copy()
        node_grid = self.get_vis_grid_pose(pos, lv)
        # vis_map = cv2.circle(vis_map, node_grid, 10, (0, 255, 0), -1)
        vis_map = cv2.rectangle(vis_map, (node_grid[0] - 8, node_grid[1] - 8), (node_grid[0] + 8, node_grid[1] + 8),
                                color, -1)
        return vis_map

    def save_viewpoint_on_topdown_map(self, save_dir=None, vis_map=None, bias_position=None, curr_position=None, curr_goal_position=None, result=None):
        if vis_map is None:
            vis_map = self.goal_map[self.curr_level][self.goal_obj_names[self.goal_class_idx]].copy()

        # for pos in self.env_class_goal_view_point[self.goal_obj_names[self.goal_class_idx]]:
        #     ## view points in dataset ##
        #     node_grid = self.get_vis_grid_pose(pos)
        #     # vis_map = cv2.circle(vis_map, node_grid, 3, (0, 255, 0), -1)
        #     shapes = np.zeros_like(vis_map, np.uint8)
        #     shapes = cv2.circle(shapes, node_grid, 3, (255, 180, 0), -1)
        #     alpha = 0.7
        #     # mask = shapes.astype(bool)
        #     mask = np.repeat(np.sum(shapes, axis=2).astype(bool)[:,:,np.newaxis], 3, axis=2)
        #     vis_map[mask] = cv2.addWeighted(vis_map, alpha, shapes, 1 - alpha, 0)[mask]
        #
        # for pos in np.array([obj['position'] for obj in self.env_goal_obj_info if obj['category'] == self.goal_class_idx]):
        #     ## goal object positions in dataset
        #     node_grid = self.get_vis_grid_pose(pos)
        #     vis_map = cv2.circle(vis_map, node_grid, 10, (255, 180, 0), -1)

        mask = np.repeat(np.sum(self.cur_graph_map, axis=2).astype(bool)[:,:,np.newaxis], 3, axis=2)
        vis_map[mask] = cv2.addWeighted(vis_map, 0.0, self.cur_graph_map, 1.0, 0)[mask]

        if curr_position is not None:
            node_grid = self.get_vis_grid_pose(curr_position + bias_position, self.curr_level)
            vis_map = cv2.rectangle(vis_map, (node_grid[0] - 8, node_grid[1] - 8), (node_grid[0] + 8, node_grid[1] + 8),
                                    (255, 0, 0), -1)
        if curr_goal_position is not None:
            node_grid = self.get_vis_grid_pose(curr_goal_position + bias_position, self.curr_level)
            vis_map = cv2.rectangle(vis_map, (node_grid[0] - 8, node_grid[1] - 8), (node_grid[0] + 8, node_grid[1] + 8),
                                    (255, 255, 0), -1)

        if result is not None:
            success = 'SUCCESS' if result['success'] == 1 else 'FAIL'
            txt = 'goal: {}, {}, SPL: {:.4f}, min_dist_to_viewpoint {:.4f}, min_dist_to_object_center {:.4f}, actions {}'.format(result['goal object'], success,
                                                                                  result['spl'],
                                                                                  result['min_dist_to_viewpoint'],
                                                                                  result['min_dist_to_goal_center'],
                                                                                  result['action step'])

            txt1 = 'goal: {}, {}, SPL: {:.4f}, actions {}'.format(
                result['goal object'], success,
                result['spl'],
                result['action step'])
            txt2 = 'viewpoint dist {:.4f}, obj center dist {:.4f}'.format(
                result['min_dist_to_viewpoint'],
                result['min_dist_to_goal_center'])


            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            color = (255, 255, 255)
            thickness = 2
            text_size = cv2.getTextSize(txt, font, font_scale, thickness)[0]
            text_position1 = (10, vis_map.shape[0] + text_size[1] * 2 + 10)
            text_position2 = (10, vis_map.shape[0] + text_size[1] * 3 + 5 + 10)

            canvas_height = vis_map.shape[0] + text_size[1] * 3 + 20
            canvas_width = vis_map.shape[1]
            canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
            canvas[:vis_map.shape[0], :] = vis_map
            canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
            cv2.putText(canvas, txt1, text_position1, font, font_scale, color, thickness)
            cv2.putText(canvas, txt2, text_position2, font, font_scale, color, thickness)
            if isinstance(save_dir, list):
                for dir in save_dir:
                    cv2.imwrite(dir, canvas)
            else:
                cv2.imwrite(save_dir, canvas)
        else:
            plt.imsave(save_dir, vis_map)

        return


    def set_level_range(self):
        self.level_range = []
        for level in self._sim.semantic_scene.levels:
            self.level_range.append([
                level.aabb.center[1] - level.aabb.sizes[1] / 2,
                level.aabb.center[1] + level.aabb.sizes[1] / 2,
            ])

        return

    def get_bbox_from_pos_size(self, position, size, lv):
        bbox = np.array([
            [position[0] - size[0] / 2, position[2] - size[2] / 2],
            [position[0] + size[0] / 2, position[2] + size[2] / 2]
        ])
        grid_bbox = np.array([
            self.get_vis_grid_pose(bbox[0], lv),
            self.get_vis_grid_pose(bbox[1], lv)
        ])
        return grid_bbox

    # def vis_topdown_obj_map(self, vis_map, obj_category):
    #     for obj in self.env_obj_info:
    #         agent_level = self.check_position2level(self.scene_height)
    #         if obj['category'] == obj_category and agent_level == obj['level']:
    #             grid_bbox = self.get_bbox_from_pos_size(obj['position'], obj['sizes'])
    #             grid_bbox = grid_bbox.tolist()
    #             shapes = np.zeros_like(vis_map, np.uint8)
    #             cv2.rectangle(shapes, tuple(grid_bbox[0]), tuple(grid_bbox[1]), (0, 0, 128), -1)
    #             alpha = 0.3
    #             mask = shapes.astype(bool)
    #             vis_map[mask] = cv2.addWeighted(vis_map, alpha, shapes, 1 - alpha, 0)[mask]
    #
    #     return vis_map
    #
    # def vis_topdown_goal_obj_map(self, vis_map, goal_obj_category):
    #     for obj in self.env_goal_obj_info:
    #         agent_level = self.check_position2level(self.scene_height)
    #         if obj['category'] == goal_obj_category and agent_level == obj['level']:
    #             grid_bbox = self.get_bbox_from_pos_size(obj['position'], obj['sizes'])
    #             grid_bbox = grid_bbox.tolist()
    #             shapes = np.zeros_like(vis_map, np.uint8)
    #             cv2.rectangle(shapes, tuple(grid_bbox[0]), tuple(grid_bbox[1]), (255, 128, 0), -1)
    #             alpha = 0.3
    #             # mask = shapes.astype(bool)
    #             mask = np.repeat(np.sum(shapes, axis=2).astype(bool)[:, :, np.newaxis], 3, axis=2)
    #             vis_map[mask] = cv2.addWeighted(vis_map, alpha, shapes, 1-alpha, 0)[mask]
    #             # vis_map = cv2.rectangle(vis_map, tuple(grid_bbox[0]), tuple(grid_bbox[1]), (255, 0, 0), -1)
    #     return vis_map


    def vis_topdown_map_with_captions(self, graph_map, curr_node=None, curr_goal_node=None,
                                      bias_position=None, curr_position=None, curr_goal_position=None,
                                      vis_goal_obj_score=None, vis_obj=None,
                                      visited_positions=None):
        # vis_map = self.map.copy()
        vis_map = self.cur_graph_map
        # if vis_obj is not None:
        #     vis_map = self.vis_topdown_obj_map(vis_map, vis_obj)

        # if vis_goal_obj_score is not None:
        #     vis_goal_obj_score = self.goal_class_idx
        #     vis_map = self.vis_topdown_goal_obj_map(vis_map, vis_goal_obj_score)

        curr_node_id, curr_goal_node_id = None, None
        if curr_node is not None:
            curr_node_id = curr_node.nodeid
        if curr_goal_node is not None:
            curr_goal_node_id = curr_goal_node.nodeid
        vis_map = self.vis_topdown_graph_map(vis_map, graph_map, vis_obj_score=vis_goal_obj_score,
                                                curr_node_id=curr_node_id, curr_goal_node_id=curr_goal_node_id,
                                                bias_position=bias_position,
                                                # curr_goal_position=curr_goal_position,
                                                visited_positions=visited_positions)

        self.cur_graph_map = vis_map
        # mask = self.cur_graph_map.astype(bool)
        mask = np.repeat(np.sum(self.cur_graph_map, axis=2).astype(bool)[:,:,np.newaxis], 3, axis=2)
        self.base_map[mask] = cv2.addWeighted(self.base_map, 0., self.cur_graph_map, 1.0, 0)[mask]

        if curr_position is not None:
            vis_map = self.vis_pos_on_topdown_map(curr_position + bias_position, self.curr_level, self.base_map)
        if curr_goal_position is not None:
            vis_map = self.vis_pos_on_topdown_map(curr_goal_position + bias_position, self.curr_level, vis_map, color=(255, 255, 0))



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

        # for i, level in enumerate(self.level_range):
        #     if pos > level[0] and pos < level[1]:
        #         return str(i)

        floor_heights = np.array([level[0] for level in self.level_range])
        floor = np.argmin(np.abs(floor_heights - pos))
        return str(floor)


    def check_goal_point_validity(self, start_position, goal_position, is_goal_obj=False):

        path = habitat_sim.ShortestPath()
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

    # def dist_to_objs(self, pos):
    #     dist = np.full(len(self.goal_obj_names), np.inf)
    #     is_valid_point = True
    #     # cur_goal_info = self._sim.semantic_scene.objects[int(self.goal_info['id'].split('_')[-1])]
    #     for i, obj in enumerate(self.env_goal_obj_info):
    #
    #         is_valid, geo_dist = self.check_goal_point_validity(pos, obj['position'], is_goal_obj=True)
    #         if not is_valid: pass
    #
    #         if geo_dist < dist[obj['category']]:
    #             dist[obj['category']] = geo_dist
    #             if obj['category'] == self.goal_class_idx:
    #                 cur_goal_info = self._sim.semantic_scene.objects[obj['id']]
    #                 # self.update_goal_info(goal_info)
    #
    #
    #     if np.sum(dist != np.full(len(self.goal_obj_names), np.inf)) == 0:
    #         is_valid_point = False
    #     self.update_goal_info(cur_goal_info)
    #
    #     return dist, is_valid_point

    def dist_to_objs(self, pos):
        dist = np.full(len(self.goal_obj_names), np.inf)
        is_valid_point = True

        # for i in range(len(self.goal_obj_names)):
        dist[self.goal_class_idx] = self.get_geodesic_distance_to_object_category(pos, self.goal_info['category'],
                                                                                  target_view_points=self.goal_info['view_points'])



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
                view_points = [v['agent_state']['position'] for v in
                               self.goal_id_to_viewpoints[int(goal_info.id.split('_')[-1])]]
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

        width_bias = int(1/(self.rot_num*2) * pw)
        width_half =int(ph/2)

        # split the panorama into 12 square images with even angles
        dirc_imgs = []
        for i in range(num_imgs):
            angle = i * 360 / num_imgs
            x = int(pw * (angle / 360)) + width_bias
            start_w = x - width_half
            end_w = x + width_half

            if start_w < 0:
                dirc_img = np.concatenate((pano_img[:, start_w:], pano_img[:, :end_w]), axis=1)
            elif end_w > self.pano_width:
                dirc_img = np.concatenate((pano_img[:, start_w:], pano_img[:, :end_w - self.pano_width]), axis=1)
            else:
                dirc_img = pano_img[:, start_w:end_w]

            # dirc_img = pano_img[:, x:x + ph]
            # if x + ph > pw:
            #     dirc_img = np.concatenate((dirc_img, pano_img[:, :x + ph - pw]), axis=1)
            dirc_imgs.append(dirc_img)
        return np.array(dirc_imgs)

    def get_cand_node_dirc(self, rgb, depth, pos, rot, vis_pos=None):
        ## rot is rotation vector
        cur_heading_idx = int(np.round(-rot[1] * 180 / np.pi / self.cand_rot_angle)) % self.rot_num
        cand_nodes = []
        cand_angle = [-30, 0, 30]
        splited_imgs = [
            rgb[:, :int(self.vo_width/2),:3],
            rgb[:, int(self.vo_width/4):int(self.vo_width*3/4),:3],
            rgb[:, int(self.vo_width/2):,:3]
        ]


        self.local_mapper.reset_map()
        depth_cm = depth * 100
        pose_origin_for_map = (pos[0], pos[2], 0)  # (x, y, o)
        pose_for_map = (pos[0], pos[2], rot[1])  # (x, y, o)
        pose_on_map_cm = self.local_mapper.get_mapper_pose_from_sim_pose(pose_for_map, pose_origin_for_map)
        pose_on_map = self.local_mapper.get_map_grid_from_sim_pose_cm(pose_on_map_cm)

        ### get current local map ###
        curr_local_map, curr_exp_map, _ = self.local_mapper.update_map(depth_cm, pose_on_map_cm)
        curr_local_map = (skimage.morphology.binary_dilation(
            curr_local_map, skimage.morphology.disk(2)
        )== True).astype(float)

        # text = goal_info['category_place']
        rot_axis = np.array([0, 1, 0])
        # head = -quaternion.as_rotation_vector(rot)[1] * 180 / np.pi

        # for global coordinate
        # turn left = positive angle
        # free cand angle idx --> right side is positive
        free_cand_nodes = np.zeros(12)
        angle_bias = np.where(self.cand_angle == -30)[0][0]
        cand_split_images = []
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
                cand_node_info = {'position': cand_pos, 'rotation': cand_rot, 'heading_idx': cur_heading_idx,
                                  'pose_on_map': cand_pose_on_grid_map, 'cand_edge': []}

                # if self.vis_floorplan:
                vis_rot_vec = rot_vec + self.abs_init_rotation
                vis_unit_vec = -np.array([np.sin(vis_rot_vec[1]), 0, np.cos(vis_rot_vec[1])])
                vis_cand_pos = vis_pos + vis_unit_vec * self.edge_range
                cand_node_info['vis_position'] = vis_cand_pos

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
                #     vis_next_pos = vis_pos + vis_unit_vec * self.edge_range * 2
                #     cand_node_info['next_node']['vis_position'] = vis_next_pos
                #
                # else:
                #     cand_node_info['next_node'] = None
                cand_node_info['next_node'] = None

                cand_nodes.append(cand_node_info)
                cand_split_images.append(splited_imgs[i])

                free_cand_nodes[angle_bias + i] = 1





        valid_cand_nodes = []
        if len(cand_split_images) > 0:
            cand_image_feat = self.common_sense_model.clip.get_image_feat(cand_split_images)
            for i in range(len(cand_nodes)):
                cand_nodes[i]['clip_feat'] = cand_image_feat[i]

                for j in range(i+1, len(cand_nodes)):
                    if self.local_mapper.is_traversable(curr_local_map, cand_nodes[i]['pose_on_map'], cand_nodes[j]['pose_on_map']):
                        cand_nodes[i]['cand_edge'].append(j)

                valid_cand_nodes.append(cand_nodes[i])

        return valid_cand_nodes

    def update_cand_node_to_graph(self, cur_node, cand_nodes):
        if len(cand_nodes) == 0:
            return
        cand_node_list = []
        for cand_node_info in cand_nodes:
            cand_node, add_new_node = self.graph_map.add_single_node(cand_node_info['position'])
            self.graph_map.update_node_goal_category(cand_node, self.goal_class_onehot)
            self.graph_map.update_node_clip_feat(cand_node, cand_node_info['clip_feat'], cand_node_info['heading_idx'])
            # self.graph_map.update_node_vis_feat(cand_node)

            torch.set_num_threads(1)
            if self.cm_type == 'comet':
                goal_cm_scores, _ = self.common_sense_model.text_image_score(self.goal_place_text_feat,
                                                                             cand_node_info['clip_feat'], feat=True,
                                                                             return_only_max=False)
                goal_cm_scores = goal_cm_scores * 0.01
                cand_node.update_goal_cm_scores(goal_cm_scores, cand_node_info['heading_idx'])


            elif self.cm_type == 'mp3d':
                goal_cm_scores, _ = self.common_sense_model.text_image_score(self.cand_place_text_feat,
                                                                             cand_node_info['clip_feat'], feat=True,
                                                                             return_only_max=False)
                goal_cm_scores = goal_cm_scores[:, :5]

                goal_cm_scores = np.round(np.max(np.exp(goal_cm_scores) / np.sum(np.exp(goal_cm_scores)), axis=1), 5)
                weighted_goal_cm_scores = goal_cm_scores * self.cand_category_room_score[self.goal_info['category']][
                                                           :5]  ## weighted by room category
                cand_node.update_goal_cm_scores(weighted_goal_cm_scores, cand_node_info['heading_idx'])
            self.graph_map.update_node_feat(cand_node)

            if add_new_node:
                curr_dist_to_objs, curr_is_valid = self.dist_to_objs(cand_node_info['vis_position'] + self.abs_init_position)
                self.graph_map.update_node_dist_to_objs(cand_node, curr_dist_to_objs)
                # if self.vis_floorplan:
                cand_node.vis_pos = cand_node_info['vis_position']

            self.graph_map.add_edge(cur_node, cand_node)
            cand_node_list.append(cand_node)

            # ## -- one step further -- ##
            # if cand_node_info['next_node'] is not None:
            #     next_node, add_new_node = self.graph_map.add_single_node(cand_node_info['next_node']['position'])
            #     self.graph_map.update_node_goal_category(next_node, self.goal_class_onehot)
            #     self.graph_map.update_node_clip_feat(next_node, cand_node_info['next_node']['clip_feat'],
            #                                          cand_node_info['next_node']['heading_idx'])
            #     # self.graph_map.update_node_vis_feat(next_node)
            #
            #     torch.set_num_threads(1)
            #     if self.cm_type == 'comet':
            #         next_node.update_goal_cm_scores(goal_cm_scores, cand_node_info['heading_idx'])
            #
            #     elif self.cm_type == 'mp3d':
            #         next_node.update_goal_cm_scores(weighted_goal_cm_scores, cand_node_info['heading_idx'])
            #     # self.graph_map.update_node_feat(cand_node)
            #
            #     if add_new_node:
            #         curr_dist_to_objs, curr_is_valid = self.dist_to_objs(
            #             cand_node_info['next_node']['vis_position'] + self.abs_init_position)
            #         self.graph_map.update_node_dist_to_objs(next_node, curr_dist_to_objs)
            #         next_node.vis_pos = cand_node_info['next_node']['vis_position']
            #
            #     self.graph_map.add_edge(cand_node, next_node)

        for i, node in enumerate(cand_node_list):
            for j in cand_nodes[i]['cand_edge']:
                self.graph_map.add_edge(node, cand_node_list[j])



    def get_value_graph(self):
        nodes = [self.graph_map.node_by_id[id] for id in self.graph_map.node_by_id.keys()]
        graph_size = len(nodes)

        node_cm_scores = torch.zeros([graph_size, 12 * 10], dtype=torch.float)
        node_features = torch.zeros([graph_size, 12 * self.args.vis_feat_dim], dtype=torch.float)
        node_goal_features = torch.zeros([graph_size, self.args.vis_feat_dim], dtype=torch.float)
        node_info_features = torch.zeros([graph_size, 1 + 3 + 12 * 10], dtype=torch.float)

        # for i in range(graph_size):
        #     node_cm_scores[i] = torch.Tensor([nodes[i].cm_score])
        # softmax_node_cm_scores = torch.softmax(node_cm_scores, dim=0)

        for i in range(graph_size):
            node_features[i] = torch.reshape(nodes[i].clip_feat, [-1])
            node_goal_features[i] = self.goal_category_feat[self.goal_class_idx]
            node_cm_scores[i] = torch.reshape(nodes[i].goal_cm_scores, [-1])
            node_info_features[i] = torch.cat([nodes[i].visited,
                                               torch.Tensor(nodes[i].pos),
                                               node_cm_scores[i]], dim=0)

        adj_mtx = torch.Tensor(self.graph_map.adj_mtx) + torch.eye(graph_size)

        node_features, node_goal_features, node_info_features, adj_mtx = \
            node_features.to(f'cuda:{self.args.model_gpu}'), \
            node_goal_features.to(f'cuda:{self.args.model_gpu}'), \
            node_info_features.to(f'cuda:{self.args.model_gpu}'), \
            adj_mtx.to(f'cuda:{self.args.model_gpu}')

        object_value = self.value_model(node_features, node_goal_features, node_info_features, adj_mtx)
        object_value = object_value.cpu().detach().numpy()

        for i, node in enumerate(nodes):
            node.pred_value = object_value[i]

        return object_value


    def get_next_subgoal_using_graph(self, cur_node):
        max_score = 0
        min_dist = 9999
        cand_node = None
        max_dist = 30

        ids = []
        cm_scores = []
        dist_scores = []
        obj_scores = []
        true_score = []
        object_value = self.get_value_graph()

        for i, id in enumerate(self.graph_map.candidate_node_ids):
            node = self.graph_map.get_node_by_id(id)
            ids.append(id)

            cm_scores.append(node.cm_score)
            obj_scores.append(np.squeeze(node.pred_value))

            # dist score
            temp_path, temp_path_length = self.get_shortest_path(cur_node.nodeid, id, self.graph_map.adj_mtx)
            if len(temp_path) == 0:
                dist_score = -10
            else:
                dist_score = max(1 - temp_path_length / max_dist, 0)
            dist_scores.append(dist_score)


            true_score.append(node.dist_to_objs[self.goal_class_idx])
            # oracle score
            if self.use_oracle:
                if node.dist_to_objs[self.goal_class_idx] < min_dist:
                    min_dist = node.dist_to_objs[self.goal_class_idx]
                    cand_node = node



        if self.use_oracle:
            return cand_node
        else:
            dist_scores = np.array(dist_scores)
            # cm_scores = np.array(cm_scores)
            # softmax_cm_scores = np.exp(cm_scores) / np.sum(np.exp(cm_scores))
            # combined_score = 0.5 * softmax_cm_scores + 0.5 * dist_scores

            obj_scores = np.array(obj_scores)
            combined_score = obj_scores + 0.1 * dist_scores
            # combined_score = obj_scores
            node_idx = np.argmax(combined_score)
            cand_node = self.graph_map.get_node_by_id(ids[node_idx])

            return cand_node


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
        if self.goal_cat == 'mp3d':
            obj_min_dist = 9999
            obj_min_pixel = None
            closest_obj_id = None
            rgb = rgb[:, :, :3]
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
                    # temp_min_dist = np.min(depth[np.nonzero(depth*masks[i])])
                    nonzero_pixel = depth[np.nonzero(depth * masks[i])]
                    temp_med_dist = np.sort(nonzero_pixel)[int(len(nonzero_pixel) / 2)]

                    if temp_med_dist < obj_min_dist:
                        obj_min_dist = temp_med_dist
                        obj_min_pixel = np.argwhere(depth * masks[i] == temp_med_dist)[0]
                        closest_obj_id = i

                        ## get position from pixel

            det_out = {
                'pred_classes': pred_classes,
                'scores': scores,
                'pred_out': pred_out,
                'masks': masks,
                'boxes': boxes,
                'closest_obj_id': closest_obj_id,
                'obj_min_dist': obj_min_dist,
                'obj_min_pixel': obj_min_pixel
            }
            if vis:
                det_out['det_img'] = img
            return det_out, obj_min_dist

        elif self.goal_cat == 'mp3d_21':
            rgb = rgb[:, :, :3]

            in_rgb = np.transpose(rgb, (2, 0, 1))
            in_depth = np.expand_dims(depth, axis=0)

            in_rgb, in_depth = torch.from_numpy(in_rgb).float().to(f"cuda:{self.args.model_gpu}"), torch.from_numpy(in_depth).float().to(f"cuda:{self.args.model_gpu}")
            in_rgb, in_depth = in_rgb.unsqueeze(0), in_depth.unsqueeze(0)
            pred = self.detector.get_predictions(in_rgb, in_depth)[0]

            pred_mask = pred[self.goal_class_idx].cpu().numpy().astype(np.uint8)
            depth_mask = (depth < self.last_mile_range).astype(np.uint8)
            mask = pred_mask * depth_mask
            nonzero_pixel = depth[np.nonzero(depth * mask)]

            if len(nonzero_pixel) > 0:
                temp_med_dist = np.sort(nonzero_pixel)[int(len(nonzero_pixel) / 2)]
                obj_min_dist = temp_med_dist
                obj_min_pixel = np.argwhere(depth * mask == temp_med_dist)[0]
            else:
                obj_min_dist = 9999
                obj_min_pixel = None

            det_out = {
                'obj_min_dist': obj_min_dist,
                'obj_min_pixel': obj_min_pixel
            }
            if vis:
                det_img = self.detector.visualize_rednet_pred(pred)
                alpha = 0.3
                mask = np.repeat(np.sum(det_img, axis=2).astype(bool)[:,:,np.newaxis], 3, axis=2)
                rgb[mask] = cv2.addWeighted(rgb, alpha, det_img, 1 - alpha, 0)[mask]

                det_out['det_img'] = rgb
            return det_out, obj_min_dist



    def get_position_from_pixel(self, cur_position, cur_rotation, depth, pixel):


        width, height = np.shape(depth)[1], np.shape(depth)[0]
        aspect_ratio = float(width) / float(height)
        fov = np.deg2rad(self.vo_hfov)
        f = width / 2.0 / np.tan(fov / 2.0)
        # fy = fx / aspect_ratio
        cy, cx = width / 2.0, height / 2.0

        z = depth[pixel[0], pixel[1]]
        x = (pixel[0] - cx) * z / f   # pixel height, pixel value up --> vis down
        y = (pixel[1] - cy) * z / f   # pixel width, pixel value up --> vis right

        rel_position = (y, x, -z)
        rot = R.from_rotvec(cur_rotation)
        rot.as_matrix()
        target_position = cur_position + rot.apply(rel_position)

        return target_position











    def get_shortest_path(self, start_node_id, end_node_id, adj_mtx):
        start_node_id, end_node_id = int(start_node_id), int(end_node_id)

        graph = csr_matrix(adj_mtx)
        dist_matrix, predecessors = shortest_path(csgraph=graph, directed=False, return_predecessors=True)
        path = [end_node_id]
        while path[-1] != start_node_id:
            path.append(predecessors[start_node_id, path[-1]])
            if path[-1] == -9999:
                return []
        path.reverse()

        path = [str(i) for i in path]
        path_length = 0
        for i, nodeid in enumerate(path[:-1]):
            path_length += self.graph_map.adj_mtx[int(nodeid)][int(path[i + 1])]

        return path[1:], path_length



    def do_explicit_action(self, cur_node, action, curr_goal_node=None):
        abs_prev_position = self._sim.agents[0].get_state().position
        obs = self._sim.step(action)
        self.abs_position = self._sim.agents[0].get_state().position
        self.path_length += self.dist_euclidean_floor(abs_prev_position, self.abs_position)
        self.abs_rotation = quaternion.as_rotation_vector(self._sim.agents[0].get_state().rotation)
        # curr_position = self.abs_position - self.abs_init_position

        curr_rotation = self.abs_rotation - self.abs_init_rotation
        # curr_rotation = quaternion.as_rotation_vector(curr_rotation) - self.abs_init_rotation
        rot = R.from_rotvec(-self.abs_init_rotation)
        curr_position_diff = rot.apply(self.abs_position - abs_prev_position)
        curr_position = self.cur_position + curr_position_diff
        self.cur_position = curr_position
        self.cur_rotation = curr_rotation




        pano_obs = self.panoramic_obs(obs, semantic=True)
        self.pano_rgb_list.append(pano_obs['rgb_panoramic'])
        self.rgb_list.append(obs['color_sensor'])
        self.depth_list.append(obs['depth_sensor'])

        det, det_dist = self.check_close_goal_det(obs['color_sensor'], obs['depth_sensor'], vis=True)

        if det_dist < self.last_mile_range:
            obs_goal_position = self.get_position_from_pixel(curr_position, curr_rotation, obs['depth_sensor'],
                                                             det['obj_min_pixel'])
            curr_state = self._sim.agents[0].get_state()
            vis_obs_goal_position = self.get_position_from_pixel(curr_state.position - self.abs_init_position,
                                                                 quaternion.as_rotation_vector(curr_state.rotation),
                                                                 obs['depth_sensor'],
                                                                 det['obj_min_pixel'])
            new_goal_obj_det = True
            for cand_goal_idx in range(len(self.goal_obs_consistency['position'])):
                if np.linalg.norm(self.goal_obs_consistency['position'][cand_goal_idx] - obs_goal_position) < 1.0:
                    self.goal_obs_consistency['count'][cand_goal_idx] += 1
                    # self.goal_obs_consistency['position'][cand_goal_idx] = obs_goal_position
                    self.goal_obs_consistency['position'][cand_goal_idx] = \
                        (self.goal_obs_consistency['position'][cand_goal_idx] * (
                                    self.goal_obs_consistency['count'][cand_goal_idx] - 1) + obs_goal_position) / \
                        self.goal_obs_consistency['count'][cand_goal_idx]
                    self.goal_obs_consistency['vis_position'][cand_goal_idx] = \
                        (self.goal_obs_consistency['vis_position'][cand_goal_idx] * (
                                self.goal_obs_consistency['count'][cand_goal_idx] - 1) + vis_obs_goal_position) / \
                        self.goal_obs_consistency['count'][cand_goal_idx]
                    new_goal_obj_det = False
                    break
            if new_goal_obj_det:
                self.goal_obs_consistency['position'].append(obs_goal_position)
                self.goal_obs_consistency['vis_position'].append(vis_obs_goal_position)
                self.goal_obs_consistency['count'].append(1)

        if self.vis_floorplan:
            self.abs_position = self._sim.agents[0].get_state().position
            self.visited_positions.append(np.array(self.abs_position) - np.array(self.abs_init_position))
            vis_graph_map = self.vis_topdown_map_with_captions(self.graph_map,
                                                               curr_node=self.cur_node,
                                                               bias_position=self.abs_init_position,
                                                               curr_position=np.array(self.abs_position) - np.array(self.abs_init_position),
                                                               curr_goal_node=curr_goal_node,
                                                               visited_positions=self.visited_positions)
            # vis_local_map = self.local_agent.get_observed_colored_map(gt=True)
            # vis_local_map = np.zeros([241, 241, 3])
            self.local_agent.reset_with_curr_pose(curr_position, curr_rotation)
            delta_dist, delta_rot = get_relative_location(curr_position, curr_rotation, curr_position)
            self.local_agent.update_gt_local_map(obs['depth_sensor'])
            self.local_agent.set_goal(delta_dist, delta_rot)
            vis_local_map = self.local_agent.get_observed_colored_map(gt=True)

            self.vis_info['cur_position'] = curr_position
            self.vis_info['mode'] = 'node search'

            # det_pano, _, _, _, _, _ = self.detector.predicted_img(self.pano_rgb_list[-1].astype(np.uint8), show=True)
            total_frame = self.make_total_frame(det['det_img'], obs['depth_sensor'], vis_graph_map, vis_local_map,
                                                pano_rgb=self.pano_rgb_list[-1],
                                                info=self.vis_info)
            self.vis_traj.append(total_frame)

        self.action_step += 1

        ## update candidate node
        cand_nodes = self.get_cand_node_dirc(self.rgb_list[-1],
                                             self.depth_list[-1], curr_position, curr_rotation, vis_pos=np.array(self.abs_position) - np.array(self.abs_init_position))
        self.update_cand_node_to_graph(cur_node, cand_nodes)

        # self.check_pano_goal_det(pano_obs['rgb_panoramic'], self.cur_node, curr_position, curr_rotation, vis=True)
        curr_position, curr_rotation = self.cur_position, self.cur_rotation
        if self.end_episode:
            return self.end_episode

        # if not action == 'move_forward':
        #     ### -- update current position -- ###
        #     self.cur_position = np.array(cur_node.pos)

        self.end_episode = False
        if len(self.goal_obs_consistency['count']) > 0:
            if np.max(self.goal_obs_consistency['count']) >= self.goal_obs_consistency_th:
                self.last_mile_navigation(obs)
                self.end_episode = True
                return self.end_episode

        return self.end_episode

    def do_panoramic_action(self, cur_node):
        action = 'turn_left'
        for i in range(int(360/self.act_rot)):
            self.end_episode = self.do_explicit_action(cur_node, action)
            if self.end_episode:
                break
        return



    def success_evaluation(self, last_position, data):
        # nearest_goal = self.find_nearest_goal(last_position, goal_class_idx)
        # goal_view_points = self.goal_id_to_viewpoints[nearest_goal['id']]
        # success = self.check_close_viewpoint(last_position, goal_view_points, th=0.1)

        viewpoint_dist = np.linalg.norm(self.env_class_goal_view_point[data['object_category']] - last_position, axis=1)
        # dist = self.get_geodesic_distance_to_object_category(last_position, data['object_category'])
        goal_list = np.array([obj['position'] for obj in self.env_goal_obj_info if obj['category'] == self.goal_class_idx])
        goal_dist = np.linalg.norm(goal_list - last_position, axis=1)
        if np.min(viewpoint_dist) < 0.1 or np.min(goal_dist) < 1.0:
            success = 1.
        else:
            success = 0.
            # dist = np.linalg.norm(self.viewpoint_goal_position - last_position)
            # if np.min(dist) < 0.1:
            #     success = 1.
            # else:
            #     success = 0.



        spl = success * data['info']['geodesic_distance'] / max(self.path_length, data['info']['geodesic_distance'])


        return success, spl, np.min(viewpoint_dist), np.min(goal_dist)

    def do_time_steps(self, data_idx):
        self.abs_position = self._sim.agents[0].get_state().position
        self.abs_rotation = quaternion.as_rotation_vector(self._sim.agents[0].get_state().rotation)
        self.abs_heading = -self.abs_rotation[1] * 180 / np.pi - self.abs_init_heading


        self.abs_heading = self.abs_heading % 360
        if self.abs_heading > 180:
            self.abs_heading -= 180
        if self.abs_heading < 180:
            self.abs_heading += 180

        # curr_position = self.abs_position - self.abs_init_position
        # curr_rotation = self.abs_rotation - self.abs_init_rotation
        # curr_heading = self.abs_heading - self.abs_init_heading

        curr_position = self.cur_position
        curr_rotation = self.cur_rotation


        max_action_step = False
        last_mile_navi_mode = False
        last_mile_obs = None
        last_mile_det = None
        invalid_edge = False

        arrive_node = False

        while True:
            if self.end_episode:
                return
            if len(self.graph_map.candidate_node_ids) <= 2:
                self.do_panoramic_action(self.cur_node)
            if invalid_edge:
                # return to the previous node
                temp_goal_node = self.graph_map.get_node_by_id(self.cur_node.nodeid)
                temp_goal_position = self.cur_node.pos
            else:
                subgoal_node = self.get_next_subgoal_using_graph(self.cur_node)
                if subgoal_node == None:
                    return

                # ## check if the subgoal is reachable
                # subgoal_position = subgoal_node.pos
                # if np.linalg.norm(subgoal_position - curr_position) < self.graph_map.max_edge_length and \
                #     self.graph_map.adj_mtx[int(self.cur_node.nodeid), int(subgoal_node.nodeid)] == 0:
                #     rot_to_face = int(np.round(((math.degrees(math.atan2(subgoal_position[0] - curr_position[0],
                #                                            -subgoal_position[2] + curr_position[2])) % 360) - (
                #                               -math.degrees(curr_rotation[1]) % 360)))) // self.act_rot
                #     if rot_to_face < -180/self.act_rot:
                #         rot_to_face += int(360/self.act_rot)
                #     elif rot_to_face > 180/self.act_rot:
                #         rot_to_face -= int(360/self.act_rot)
                #     for rot in range(abs(rot_to_face)):
                #         if rot_to_face < 0:
                #             action = 'turn_left'
                #         else:
                #             action = 'turn_right'
                #         self.end_episode = self.do_explicit_action(self.cur_node, action, curr_goal_node=subgoal_node)
                #         if self.end_episode:
                #             # last mile navigation activated in explicit action
                #             return
                #
                #     curr_position = self.cur_position
                #     curr_rotation = self.cur_rotation

                subgoal_id = subgoal_node.nodeid

                temp_path, temp_path_length = self.get_shortest_path(self.cur_node.nodeid, subgoal_id, self.graph_map.adj_mtx)

                # for node_id in temp_path:
                ## --- one node step update --- ##
                node_id = temp_path[0]
                temp_goal_node = self.graph_map.get_node_by_id(node_id)
                temp_goal_position = temp_goal_node.pos

            obs = self._sim.get_sensor_observations()
            if len(self.goal_obs_consistency['count']) > 0:
                if np.max(self.goal_obs_consistency['count']) >= self.goal_obs_consistency_th:
                    self.last_mile_navigation(obs)
                    return


            self.local_agent.reset_with_curr_pose(curr_position, curr_rotation)
            delta_dist, delta_rot = get_relative_location(curr_position, curr_rotation, temp_goal_position)
            self.local_agent.update_gt_local_map(obs['depth_sensor'])
            self.local_agent.set_goal(delta_dist, delta_rot)

            if self.vis_floorplan:
                self.abs_position = self._sim.agents[0].get_state().position
                self.visited_positions.append(np.array(self.abs_position) - np.array(self.abs_init_position))
                vis_graph_map = self.vis_topdown_map_with_captions(self.graph_map,
                                                                   curr_node=self.cur_node,
                                                                   bias_position=self.abs_init_position,
                                                                   curr_position=np.array(self.abs_position) - np.array(self.abs_init_position),
                                                                   curr_goal_node=subgoal_node,
                                                                   visited_positions=self.visited_positions)
                vis_local_map = self.local_agent.get_observed_colored_map(gt=True)
            local_action_cnt = 0
            terminate_local = False
            while self.dist_euclidean_floor(curr_position, temp_goal_position) >= self.follower_goal_radius:
                action, terminate_local = self.local_agent.navigate_local(gt=True)
                if terminate_local:
                    invalid_edge = True
                    break
                action = self.local_agent.action_idx_map[action]
                prev_position = self._sim.agents[0].get_state().position
                obs = self._sim.step(action)
                self.path_length += self.dist_euclidean_floor(prev_position, self._sim.agents[0].get_state().position)
                self.action_step += 1
                local_action_cnt += 1
                if action == 'move_forward':
                    arrive_node = False


                curr_state = self._sim.agents[0].get_state()
                curr_rotation = quaternion.as_rotation_vector(curr_state.rotation) - self.abs_init_rotation
                rot = R.from_rotvec(-self.abs_init_rotation)
                curr_position_diff = rot.apply(curr_state.position - prev_position)
                curr_position = self.cur_position + curr_position_diff
                self.cur_position = curr_position
                self.cur_rotation = curr_rotation


                det, det_dist = self.check_close_goal_det(obs['color_sensor'], obs['depth_sensor'], vis=True)

                pano_obs = self.panoramic_obs(obs, semantic=True)
                self.pano_rgb_list.append(pano_obs['rgb_panoramic'])
                # self.rgb_list.append(det['det_img'])
                self.rgb_list.append(obs['color_sensor'])
                self.depth_list.append(obs['depth_sensor'])



                self.local_agent.gt_new_sim_origin = get_sim_location(curr_position, quaternion.from_rotation_vector(curr_rotation))
                self.local_agent.update_gt_local_map(obs['depth_sensor'])

                if self.vis_floorplan:
                    self.vis_info['cur_position'] = curr_position
                    if invalid_edge:
                        self.vis_info['mode'] = 'Return to the previous node'
                    else:
                        self.vis_info['mode'] = 'Exploration'
                    vis_local_map = self.local_agent.get_observed_colored_map(gt=True)
                    # det_pano, _, _, _, _, _ = self.detector.predicted_img(self.pano_rgb_list[-1].astype(np.uint8), show=True)
                    total_frame = self.make_total_frame(det['det_img'], obs['depth_sensor'], vis_graph_map, vis_local_map,
                                                        pano_rgb=self.pano_rgb_list[-1],
                                                        info=self.vis_info)
                    self.vis_traj.append(total_frame)

                if det_dist < self.last_mile_range:
                    obs_goal_position = self.get_position_from_pixel(curr_position, curr_rotation, obs['depth_sensor'], det['obj_min_pixel'])
                    vis_obs_goal_position = self.get_position_from_pixel(curr_state.position - self.abs_init_position,
                                                                         quaternion.as_rotation_vector(
                                                                             curr_state.rotation), obs['depth_sensor'],
                                                                         det['obj_min_pixel'])
                    new_goal_obj_det = True
                    for cand_goal_idx in range(len(self.goal_obs_consistency['position'])):
                        if np.linalg.norm(self.goal_obs_consistency['position'][cand_goal_idx] - obs_goal_position) < 1.0 - self.step_size:
                            self.goal_obs_consistency['count'][cand_goal_idx] += 1
                            # self.goal_obs_consistency['position'][cand_goal_idx] = obs_goal_position
                            self.goal_obs_consistency['position'][cand_goal_idx] = \
                                (self.goal_obs_consistency['position'][cand_goal_idx] * (self.goal_obs_consistency['count'][cand_goal_idx] - 1) + obs_goal_position) / self.goal_obs_consistency['count'][cand_goal_idx]
                            self.goal_obs_consistency['vis_position'][cand_goal_idx] = \
                                (self.goal_obs_consistency['vis_position'][cand_goal_idx] * (
                                            self.goal_obs_consistency['count'][
                                                cand_goal_idx] - 1) + vis_obs_goal_position) / \
                                self.goal_obs_consistency['count'][cand_goal_idx]
                            new_goal_obj_det = False
                            break
                    if new_goal_obj_det:
                        self.goal_obs_consistency['position'].append(obs_goal_position)
                        self.goal_obs_consistency['vis_position'].append(vis_obs_goal_position)
                        self.goal_obs_consistency['count'].append(1)

                if len(self.goal_obs_consistency['count']) > 0:
                    if np.max(self.goal_obs_consistency['count']) >= self.goal_obs_consistency_th:
                        last_mile_navi_mode = True
                        last_mile_obs = obs
                        # last_mile_det = det
                        break


                if obs['collided'] or \
                        (action == 'move_forward' and np.linalg.norm(prev_position - curr_state.position) < self.step_size * 0.3):
                    self.local_agent.collision = True





                cand_nodes = self.get_cand_node_dirc(self.rgb_list[-1], self.depth_list[-1], curr_position,
                                                     curr_rotation, np.array(curr_state.position)-np.array(self.abs_init_position))
                cur_node_id, _ = self.graph_map.get_nearest_node(curr_position)
                self.update_cand_node_to_graph(self.graph_map.node_by_id[cur_node_id], cand_nodes)




                self.local_agent.gt_new_sim_origin = get_sim_location(curr_position,
                                                                      quaternion.from_rotation_vector(curr_rotation))
                self.local_agent.update_gt_local_map(obs['depth_sensor'])


                if self.vis_floorplan:
                    self.abs_position = self._sim.agents[0].get_state().position
                    self.visited_positions.append(np.array(self.abs_position) - np.array(self.abs_init_position))
                    vis_graph_map = self.vis_topdown_map_with_captions(self.graph_map,
                                                                   curr_node=self.cur_node,
                                                                   bias_position=self.abs_init_position,
                                                                   curr_position=np.array(self.abs_position) - np.array(self.abs_init_position),
                                                                   curr_goal_node=subgoal_node,
                                                                   visited_positions=self.visited_positions)
                    vis_local_map = self.local_agent.get_observed_colored_map(gt=True)


                if self.end_episode:
                    return

                if self.action_step > self.max_step:
                    max_action_step = True
                    break

                if local_action_cnt >= self.max_local_action_trial:
                    invalid_edge = True
                    break
                else:
                    invalid_edge = False



            if max_action_step:
                break
            if last_mile_navi_mode:
                break
            if local_action_cnt < self.max_local_action_trial and not terminate_local:
                invalid_edge = False
            if invalid_edge:
                self.graph_map.delete_edge(self.cur_node, temp_goal_node)
                continue


            self.cur_node = temp_goal_node
            arrive_node = True
            invalid_edge = False

            curr_state = self._sim.agents[0].get_state()
            curr_obs = self._sim.get_sensor_observations()
            splited_imgs = [
                curr_obs['color_sensor'][:, :int(self.vo_width / 2), :3],
                curr_obs['color_sensor'][:, int(self.vo_width / 4):int(self.vo_width * 3 / 4), :3],
                curr_obs['color_sensor'][:, int(self.vo_width / 2):, :3]
            ]

            splited_img_feat = self.common_sense_model.clip.get_image_feat(splited_imgs)
            cur_heading_idx = int(np.round(-curr_rotation[1] * 180 / np.pi / self.cand_rot_angle)) % self.rot_num
            self.graph_map.update_node_goal_category(self.cur_node, self.goal_class_onehot)

            for i in range(len(splited_imgs)):
                dirc_head_idx = (cur_heading_idx -1 + i) % self.rot_num
                self.graph_map.update_node_clip_feat(self.cur_node, splited_img_feat[i], dirc_head_idx)

                if self.cm_type == 'comet':
                    goal_cm_scores, _ = self.common_sense_model.text_image_score(self.goal_place_text_feat,
                                                                                 splited_img_feat[i], feat=True,
                                                                                 return_only_max=False)
                    # goal_cm_scores = torch.softmax(goal_cm_scores, dim=1)
                    goal_cm_scores = goal_cm_scores * 0.01
                    self.cur_node.update_goal_cm_scores(goal_cm_scores, dirc_head_idx)


                elif self.cm_type == 'mp3d':
                    goal_cm_scores, _ = self.common_sense_model.text_image_score(self.cand_place_text_feat,
                                                                                 splited_img_feat[i], feat=True,
                                                                                 return_only_max=False)
                    goal_cm_scores = goal_cm_scores[:, :5]

                    goal_cm_scores = np.round(np.max(np.exp(goal_cm_scores) / np.sum(np.exp(goal_cm_scores)), axis=1),
                                              5)
                    weighted_goal_cm_scores = goal_cm_scores * self.cand_category_room_score[
                                                                   self.goal_info['category']][
                                                               :5]  ## weighted by room category
                    self.cur_node.update_goal_cm_scores(weighted_goal_cm_scores, dirc_head_idx)

            self.graph_map.update_node_vis_feat(self.cur_node)

            # self.graph_map.update_node_cm_score(self.cur_node, cm_score[0] * self.goal_info['category_place_score'][arg_cm_score[0]])
            self.graph_map.update_node_visited(self.cur_node)
            self.graph_map.update_node_feat(self.cur_node)

            curr_dist_to_objs, curr_is_valid = self.dist_to_objs(curr_state.position)
            self.graph_map.update_node_dist_to_objs(self.cur_node, curr_dist_to_objs)
            # self.graph_map.update_node_room(subgoal_node, self.check_position2room(subgoal_node.pos, self.room_info))
            # self.graph_map.update_node_cand_info(subgoal_node, cand_node_info)

            cand_nodes = self.get_cand_node_dirc(self.rgb_list[-1], self.depth_list[-1], curr_position, curr_rotation, vis_pos=np.array(curr_state.position)- np.array(self.abs_init_position))
            self.update_cand_node_to_graph(self.cur_node, cand_nodes)


            # prev_position = curr_state.position
            # # self.check_pano_goal_det(curr_pano_obs['rgb_panoramic'], self.cur_node, curr_position, curr_rotation, vis=True)
            #
            #
            #
            # curr_state = self._sim.agents[0].get_state()
            # curr_rotation = quaternion.as_rotation_vector(curr_state.rotation) - self.abs_init_rotation
            # rot = R.from_rotvec(-self.abs_init_rotation)
            # curr_position_diff = rot.apply(curr_state.position - prev_position)
            # curr_position = self.cur_position + curr_position_diff
            # self.cur_position = curr_position
            # self.cur_rotation = curr_rotation


            if self.end_episode:
                return


            if last_mile_navi_mode:
                break

            # obs = self._sim.get_sensor_observations()
            # semantic = obs['semantic_sensor']

            # det, det_dist = self.check_close_goal_det(obs['color_sensor'], obs['depth_sensor'])
            # if det_dist < 0.75:
            #     return

            if max_action_step:
                break

        ### Last mile navigation ###
        if last_mile_navi_mode:
            self.last_mile_navigation(last_mile_obs)
            return
        return

    def last_mile_navigation(self, last_mile_obs):
        # curr_state = self._sim.agents[0].get_state()
        # curr_position = curr_state.position - self.abs_init_position
        # curr_rotation = quaternion.as_rotation_vector(curr_state.rotation)  # - self.abs_init_rotation
        # self.cur_position, self.cur_rotation = curr_position, curr_rotation
        curr_position = self.cur_position
        curr_rotation = self.cur_rotation


        # last_mile_start_position = curr_state.position - self.abs_init_position
        # last_mile_start_rotation = quaternion.as_rotation_vector(curr_state.rotation)  # - self.abs_init_rotation
        last_mile_start_position = self.cur_position
        last_mile_start_rotation = self.cur_rotation


        # ## get target position
        # closest_obj_id = last_mile_det['closest_obj_id']
        # masked_depth = last_mile_det['masks'][closest_obj_id] * last_mile_obs['depth_sensor']
        # nonzero_pixel = masked_depth[np.nonzero(masked_depth)]
        # target_pixel = np.argwhere(masked_depth == np.sort(nonzero_pixel)[int(len(nonzero_pixel)/2)])[0]
        # # target_pixel = np.argwhere(masked_depth == np.min(nonzero_pixel))[0]
        #
        # # target_pixel = [int(last_mile_det['boxes'][closest_obj_id][0] + last_mile_det['boxes'][closest_obj_id][2] / 2),
        # #                 int(last_mile_det['boxes'][closest_obj_id][1] + last_mile_det['boxes'][closest_obj_id][3] / 2)]
        # target_position = self.get_position_from_pixel(curr_position, curr_rotation, last_mile_obs['depth_sensor'], target_pixel)

        ## get target position from the consistency
        goal_obs_idx = np.argmax(self.goal_obs_consistency['count'])
        goal_obs_position = self.goal_obs_consistency['position'][goal_obs_idx]
        goal_obs_count = self.goal_obs_consistency['count'][goal_obs_idx]

        target_position = self.goal_obs_consistency['position'][np.argmax(self.goal_obs_consistency['count'])]
        self.object_goal_position = np.copy(target_position)
        self.vis_object_goal_position = np.copy(
            self.goal_obs_consistency['vis_position'][np.argmax(self.goal_obs_consistency['count'])])

        self.local_agent.reset_with_curr_pose(curr_position, curr_rotation)
        delta_dist, delta_rot = get_relative_location(curr_position, curr_rotation, target_position)
        self.local_agent.update_gt_local_map(last_mile_obs['depth_sensor'])
        self.local_agent.set_goal(delta_dist, delta_rot)

        object_goal_loc = np.copy(self.local_agent.goal)
        # get nearest navigable goal
        self.local_agent.goal, goal_updated = self.local_agent.get_neareset_navigable_goal(
            self.local_agent.gt_local_map,
            (self.local_agent.stg_x, self.local_agent.stg_y),
            object_goal_loc)
        if goal_updated:
            target_position = self.local_mapper.get_sim_pose_from_mapper_coords(self.local_agent.goal,
                                                                                last_mile_start_position,
                                                                                last_mile_start_rotation)
        if self.vis_floorplan:
            self.visited_positions.append(np.array(self.abs_position) - np.array(self.abs_init_position))
            # vis_graph_map = self.vis_topdown_map_with_captions(self.graph_map,
            #                                                    curr_node=self.cur_node,
            #                                                    curr_position=curr_position,
            #                                                    curr_goal_position=target_position,
            #                                                    visited_positions=self.visited_positions)
            # vis_local_map = self.local_agent.get_observed_colored_map(gt=True)


        while True:
            if self.dist_euclidean_floor(curr_position, target_position) < self.step_size or \
                    self.dist_euclidean_floor(curr_position, self.object_goal_position) < 1.0 - (self.step_size/2 +0.05):
                break


            action, terminate_local = self.local_agent.navigate_local(gt=True)
            action = self.local_agent.action_idx_map[action]
            prev_position = self._sim.agents[0].get_state().position
            obs = self._sim.step(action)
            self.path_length += self.dist_euclidean_floor(prev_position, self._sim.agents[0].get_state().position)

            det, det_dist = self.check_close_goal_det(obs['color_sensor'], obs['depth_sensor'], vis=True)

            # self.rgb_list.append(det['det_img'])
            pano_obs = self.panoramic_obs(obs, semantic=False)
            self.pano_rgb_list.append(pano_obs['rgb_panoramic'])
            self.rgb_list.append(obs['color_sensor'])
            self.depth_list.append(obs['depth_sensor'])


            # if det_dist < self.goal_det_dist:
            #     return

            # curr_state = self._sim.agents[0].get_state()
            # curr_position = curr_state.position - self.abs_init_position
            # curr_rotation = quaternion.as_rotation_vector(curr_state.rotation)  # - self.abs_init_rotation

            curr_state = self._sim.agents[0].get_state()
            curr_rotation = quaternion.as_rotation_vector(curr_state.rotation) - self.abs_init_rotation
            rot = R.from_rotvec(-self.abs_init_rotation)
            curr_position_diff = rot.apply(curr_state.position - prev_position)
            curr_position = self.cur_position + curr_position_diff
            self.cur_position = curr_position
            self.cur_rotation = curr_rotation

            if obs['collided']or \
                        (action == 'move_forward' and np.linalg.norm(prev_position - curr_state.position) < self.step_size * 0.3):
                self.local_agent.collision = True
            self.action_step += 1


            self.local_agent.gt_new_sim_origin = get_sim_location(curr_position,
                                                                  quaternion.from_rotation_vector(curr_rotation))
            self.local_agent.update_gt_local_map(obs['depth_sensor'])

            if self.vis_floorplan:
                self.abs_position = self._sim.agents[0].get_state().position
                self.abs_rotation = quaternion.as_rotation_vector(self._sim.agents[0].get_state().rotation)
                self.visited_positions.append(np.array(self.abs_position) - np.array(self.abs_init_position))
                vis_graph_map = self.vis_topdown_map_with_captions(self.graph_map,
                                                                   curr_node=self.cur_node,
                                                                   bias_position=self.abs_init_position,
                                                                   curr_position=np.array(self.abs_position) - np.array(self.abs_init_position),
                                                                   curr_goal_position=self.vis_object_goal_position,
                                                                   visited_positions=self.visited_positions)
                vis_local_map = self.local_agent.get_observed_colored_map(gt=True)

                self.vis_info['cur_position'] = curr_position
                self.vis_info['mode'] = 'Last mile'
                # det_pano, _, _, _, _, _ = self.detector.predicted_img(self.pano_rgb_list[-1].astype(np.uint8), show=True)
                total_frame = self.make_total_frame(det['det_img'], obs['depth_sensor'], vis_graph_map, vis_local_map,
                                                    pano_rgb=self.pano_rgb_list[-1],
                                                    info=self.vis_info)
                self.vis_traj.append(total_frame)

            if self.action_step > self.max_step:
                return

            if det_dist < self.last_mile_range:
                obs_goal_position = self.get_position_from_pixel(curr_position, curr_rotation, obs['depth_sensor'],
                                                                 det['obj_min_pixel'])
                vis_obs_goal_position = self.get_position_from_pixel(
                    np.array(self.abs_position) - np.array(self.abs_init_position),
                    self.abs_rotation, obs['depth_sensor'],
                    det['obj_min_pixel'])

                if np.linalg.norm(self.object_goal_position - obs_goal_position) < 1.0- self.step_size:
                    goal_obs_count += 1
                    self.object_goal_position = (self.object_goal_position * (goal_obs_count - 1) + obs_goal_position) / goal_obs_count
                    self.vis_object_goal_position = (self.vis_object_goal_position * (
                            goal_obs_count - 1) + vis_obs_goal_position) / goal_obs_count
                    delta_dist, delta_rot = get_relative_location(curr_position, curr_rotation, self.object_goal_position)
                    self.local_agent.set_goal(delta_dist, delta_rot)
                    object_goal_loc = np.copy(self.local_agent.goal)

            self.local_agent.goal, goal_updated = self.local_agent.get_neareset_navigable_goal(
                self.local_agent.gt_local_map,
                (self.local_agent.stg_x, self.local_agent.stg_y),
                object_goal_loc)
            if goal_updated:
                target_position = self.local_mapper.get_sim_pose_from_mapper_coords(self.local_agent.goal,
                                                                                    last_mile_start_position,
                                                                                    last_mile_start_rotation)




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

        self.set_level_range()

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
            self.env_class_goal_view_point = {}
            self.env_class_goal_view_point_level = {}
            for obj_name in self.goal_obj_names:
                self.env_class_goal_obj[obj_name] = []
                self.env_class_goal_view_point[obj_name] = []
                self.env_class_goal_view_point_level[obj_name] = {}
                for lv in range(len(self.level_range)):
                    self.env_class_goal_view_point_level[obj_name][str(lv)] = []

            for obj in self.obj_list:
                if obj.category.name() in self.goal_obj_names:
                    self.env_class_goal_obj[obj.category.name()].append(obj)
                    if obj.semantic_id in self.goal_id_to_viewpoints:
                        view_point_positions = [v['agent_state']['position'] for v in self.goal_id_to_viewpoints[obj.semantic_id]]
                        self.env_class_goal_view_point[obj.category.name()].extend(view_point_positions)
                        self.env_class_goal_view_point_level[obj.category.name()][obj.region.level.id].extend(view_point_positions)



            self.env_class_goal_view_point = {k:np.array(v) for k,v in self.env_class_goal_view_point.items()}



        ## set floor 0 vertical range


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
        for goal in self.dataset['episodes']:
            # if goal['info']['geodesic_distance'] < 1.0:
            #     continue
            if self.args.dataset == 'mp3d' or self.args.dataset == 'hm3d':
                if abs(goal['start_position'][1] - goal['info']['best_viewpoint_position'][1]) > 1.0:
                    continue
            if goal['object_category'] in self.goal_obj_names:
                valid_traj_list.append(goal)
                goal_traj_nums[self.goal_obj_names.index(goal['object_category'])] += 1
                goal_category = goal['object_category']


        print("Get env information done")

        self.goal_category_room, self.goal_category_room_feat, self.goal_category_room_score, \
        self.cand_category_room, self.cand_category_room_feat, self.cand_category_room_score = \
            self.init_commonsense_candidate_room(self.goal_obj_names, mp3d_room_names)

        self.goal_category_feat = self.common_sense_model.clip.get_text_feat(self.goal_obj_names).type(torch.float32)



        if len(self.env_goal_obj_info) == 0:
            self._sim.close()
            del self._sim
            return None, None


        self.env_size = abs(self._sim.semantic_scene.aabb.sizes[0] * self._sim.semantic_scene.aabb.sizes[0])
        # self.n_for_env = int(self.env_size/4)

        self.pathfinder = self._sim.pathfinder
        self.follower = habitat_sim.GreedyGeodesicFollower(self.pathfinder, self._sim.agents[0])

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





        trial = 0
        if self.data_type == 'train':
            interval = 10
        else:
            interval = 1

        max_data_num = len(valid_traj_list) // interval

        ## -- floor plan -- ##
        if self.vis_floorplan:
            self.calculate_navmesh()
            self.update_cur_floor_map()
            self.vis_obj_viewpoint_on_floormap()

            ## save floor map image
            floor_map_dir = f"{self.args.save_dir}/{self.data_type}/{self.env_name}/floor_map"
            if not os.path.exists(floor_map_dir):
                os.makedirs(floor_map_dir)
            for lv in range(len(self.map)):
                cv2.imwrite(floor_map_dir + f'/floor_map_lv{lv}.png', cv2.cvtColor(self.map[lv], cv2.COLOR_BGR2RGB))
                for obj_name in self.goal_obj_names:
                    cv2.imwrite(floor_map_dir + f'/floor_map_lv{lv}_{obj_name}.png', cv2.cvtColor(self.goal_map[lv][obj_name],cv2.COLOR_BGR2RGB))
            print("Save floor map done")





        for traj in valid_traj_list[0:len(valid_traj_list):interval]:
            data_idx += 1
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

            if self.args.dataset == 'gibson':
                selem = skimage.morphology.disk(2)
                floor_idx = traj['floor_id']
                traversible = skimage.morphology.binary_dilation(self.dataset_info[floor_idx]['sem_map'][0], selem) != True
                traversible = 1 - traversible
                planner = FMMPlanner(self.args, traversible, 360 // self.act_rot, 1)
                selem = skimage.morphology.disk(int(1.0 * 100. / self.args.map_resolution))
                goal_map = skimage.morphology.binary_dilation(self.dataset_info[floor_idx]['sem_map'][traj['object_id'] + 1], selem) != True
                goal_map = 1 - goal_map
                planner.set_multi_goal(goal_map)
                self.gt_planner = planner

                x = -traj['start_position'][2]
                y = -traj['start_position'][0]
                min_x, min_y = self.dataset_info[floor_idx]['origin'] / 100.0
                map_loc = int((-y - min_y) * 20.), int((-x - min_x) * 20.)
                shortest_path_length = self.gt_planner.fmm_dist[map_loc] / 20.0 + 1.0

            self.cur_data_idx = data_idx
            self.action_step = 0
            # self.goal_obs_consistency = [] # { 'position': np.zeros([3]), 'count': 0}
            self.goal_obs_consistency = {
                'position': [],
                'vis_position': [],
                'count': []
            }
            self.path_length = 1e-5
            self.visited_positions = []

            agent_id = self._sim_settings["default_agent"]
            agent = self._sim.initialize_agent(agent_id)
            self.local_navi_module.reset_sim_and_agent(self._sim, agent)


            start_state = agent.get_state()
            start_state.position = traj['start_position']
            start_state.rotation = traj['start_rotation']

            start_state.sensor_states = dict()
            agent.set_state(start_state)

            self.abs_init_position = start_state.position
            self.abs_init_rotation = quaternion.as_rotation_vector(start_state.rotation)
            self.abs_init_heading = -self.abs_init_rotation[1] * 180 / np.pi
            self.abs_rot = R.from_rotvec(self.abs_init_rotation)

            self.abs_position = start_state.position
            self.abs_rotation = quaternion.as_rotation_vector(start_state.rotation)
            self.abs_heading = -self.abs_rotation[1] * 180 / np.pi

            self.cur_position = np.zeros([3])
            # self.cur_rotation = quaternion.as_rotation_vector(start_state.rotation)
            self.cur_rotation = np.zeros([3])
            self.cur_heading = np.zeros([3])

            ## -- floor plan -- ##
            if self.vis_floorplan:
                self.curr_level = int(self.check_position2level(start_state.position[1]))
                self.base_map = self.map[self.curr_level].copy()
                self.cur_graph_map = np.zeros_like(self.map[self.curr_level])

            self.end_episode = False
            self.object_goal_position = None
            self.vis_object_goal_position = None
            obs = self._sim.get_sensor_observations()

            if self.args.dataset == 'mp3d' or self.args.dataset == 'hm3d':
                goal_info = self._sim.semantic_scene.objects[traj['info']['closest_goal_object_id']]
                self.goal_info = self.update_goal_info(goal_info)
                self.goal_info['goal_name'] = goal_info.category.name()
                self.goal_class_idx = self.goal_obj_names.index(goal_info.category.name())

            elif self.args.dataset == 'gibson':
                self.goal_info = {}
                self.goal_info['category'] = traj['object_category']
                # self.goal_info['goal_name'] = traj['object_category']
                self.goal_class_idx = self.goal_obj_names.index(traj['object_category'])

            self.goal_class_onehot = torch.zeros([len(self.goal_obj_names)])
            self.goal_class_onehot[self.goal_class_idx] = 1

            # self.goal_info['category_place'] = self.common_sense_model.gen_pred_words(self.goal_info['category'])
            # self.goal_place_text_feat = self.common_sense_model.clip.get_text_feat(self.goal_info['category_place']).type(torch.float32)
            self.goal_info['category_place'] = self.goal_category_room[self.goal_info['category']]
            self.goal_info['category_place_score'] = self.goal_category_room_score[self.goal_info['category']]
            self.goal_place_text_feat = self.goal_category_room_feat[self.goal_info['category']]
            self.cand_place_text_feat = self.cand_category_room_feat[self.goal_info['category']]

            ## init graph map
            self.graph_map = GraphMap(self.args)
            self.graph_map.goal_text_clip_feat = self.goal_category_feat[self.goal_class_idx]
            self.graph_map.goal_cm_info = {
                'goal_category_room': self.goal_category_room[self.goal_info['category']],
                'goal_category_room_feat': self.goal_category_room_feat[self.goal_info['category']],
                'goal_category_room_score': self.goal_category_room_score[self.goal_info['category']],
                'cand_category_room': self.cand_category_room[self.goal_info['category']],
                'cand_category_room_feat': self.cand_category_room_feat[self.goal_info['category']],
                'cand_category_room_score': self.cand_category_room_score[self.goal_info['category']],
            }

            self.graph_map.env_data = {
                'env_name': self.env_name,
                'level': self.curr_level,
                # 'floor_plan': self.base_map,
                # 'floor_plan_with_goal': self.goal_map[self.curr_level].copy(),
                'bias_position': self.abs_init_position,
                'bias_rotation': self.abs_init_rotation,
                'env_bound': self._sim.pathfinder.get_bounds()
            }

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
            det, det_dist = self.check_close_goal_det(obs['color_sensor'], obs['depth_sensor'], vis=True)


            # set initial node
            self.cur_node, _ = self.graph_map.add_single_node(self.cur_position)
            self.cur_node.vis_pos = np.array(self.abs_position) - np.array(self.abs_init_position)
            # cur_node = self.graph_map.get_node_by_pos(self.cur_position)

            # similarity, pano_clip_feat = self.common_sense_model.clip.get_text_image_sim(self.goal_info['category'], pano_images, out_img_feat=True)
            # self.graph_map.update_node_clip_feat(self.cur_node, pano_clip_feat[0])
            # value = np.round(np.max(similarity, axis=1), 3)

            self.graph_map.update_node_goal_category(self.cur_node, self.goal_class_onehot)

            torch.set_num_threads(1)
            curr_obs = self._sim.get_sensor_observations()
            splited_imgs = [
                curr_obs['color_sensor'][:, :int(self.vo_width / 2), :3],
                curr_obs['color_sensor'][:, int(self.vo_width / 4):int(self.vo_width * 3 / 4), :3],
                curr_obs['color_sensor'][:, int(self.vo_width / 2):, :3]
            ]

            splited_img_feat = self.common_sense_model.clip.get_image_feat(splited_imgs)
            cur_heading_idx = int(np.round(-self.cur_rotation[1] * 180 / np.pi / self.cand_rot_angle)) % self.rot_num
            self.graph_map.update_node_goal_category(self.cur_node, self.goal_class_onehot)

            for i in range(len(splited_imgs)):
                dirc_head_idx = (cur_heading_idx - 1 + i) % self.rot_num
                self.graph_map.update_node_clip_feat(self.cur_node, splited_img_feat[i], dirc_head_idx)
                if self.cm_type == 'comet':
                    goal_cm_scores, _ = self.common_sense_model.text_image_score(self.goal_place_text_feat,
                                                                                 splited_img_feat[i], feat=True,
                                                                                 return_only_max=False)
                    # goal_cm_scores = torch.softmax(goal_cm_scores, dim=1)
                    goal_cm_scores = goal_cm_scores * 0.01
                    self.cur_node.update_goal_cm_scores(goal_cm_scores, dirc_head_idx)


                elif self.cm_type == 'mp3d':
                    goal_cm_scores, _ = self.common_sense_model.text_image_score(self.cand_place_text_feat,
                                                                                 splited_img_feat[i], feat=True,
                                                                                 return_only_max=False)
                    goal_cm_scores = goal_cm_scores[:, :5]

                    goal_cm_scores = np.round(np.max(np.exp(goal_cm_scores) / np.sum(np.exp(goal_cm_scores)), axis=1), 5)
                    weighted_goal_cm_scores = goal_cm_scores * self.cand_category_room_score[self.goal_info['category']][:5]  ## weighted by room category
                    self.cur_node.update_goal_cm_scores(weighted_goal_cm_scores, dirc_head_idx)

            self.graph_map.update_node_vis_feat(self.cur_node)

            curr_dist_to_objs, curr_is_valid_point = self.dist_to_objs(self.abs_position)
            if not curr_is_valid_point:
                continue
            self.graph_map.update_node_dist_to_objs(self.cur_node, curr_dist_to_objs)
            self.graph_map.update_node_visited(self.cur_node)
            self.graph_map.update_node_feat(self.cur_node)
            self.graph_map.update_node_is_start(self.cur_node)


            ## update candidate node

            # cand_nodes = self.get_cand_node(self.pano_rgb_list[-1], self.cur_position, self.cur_heading, self.goal_info)
            cand_nodes = self.get_cand_node_dirc(self.rgb_list[-1], self.depth_list[-1], self.cur_position, self.cur_rotation, np.array(self.abs_position)-np.array(self.abs_init_position))
            self.update_cand_node_to_graph(self.cur_node, cand_nodes)

            if self.vis_floorplan:
                vis_graph_map = self.vis_topdown_map_with_captions(self.graph_map,
                                                                   curr_node=self.cur_node,
                                                                   bias_position=self.abs_init_position,
                                                                   # vis_goal_obj_score=self.goal_class_idx,
                                                                   )
                # vis_local_map = self.local_agent.get_observed_colored_map(gt=True)
                vis_local_map = np.zeros([241, 241, 3])
                self.vis_info['cur_position'] = self.cur_position
                self.vis_info['mode'] = 'Exploration'
                # det_pano, _, _, _, _, _ = self.detector.predicted_img(self.pano_rgb_list[-1].astype(np.uint8), show=True)
                total_frame = self.make_total_frame(det['det_img'], obs['depth_sensor'], vis_graph_map, vis_local_map,
                                                    pano_rgb=self.pano_rgb_list[-1],
                                                    info=self.vis_info)
                self.vis_traj.append(total_frame)


            data_dir = f"{self.args.save_dir}/{self.data_type}/{self.env_name}"
            if not os.path.exists(data_dir): os.makedirs(data_dir)

            self.do_panoramic_action(self.cur_node)
            self.do_time_steps(data_idx)

            # try:
            #
            #
            #     self.do_panoramic_action(self.cur_node)
            #     self.do_time_steps(data_idx)
            #
            # except KeyboardInterrupt:
            #     print("KeyboardInterrupt")
            #     break
            #
            # except:
            #     pass





            ## --- evaluation --- ##
            last_position = agent.get_state().position
            last_rotation = agent.get_state().rotation
            if self.args.dataset == 'mp3d' or self.args.dataset == 'hm3d':
                shortest_path_length = traj['info']['geodesic_distance']
                success, spl, min_dist_to_viewpoint, min_dist_to_goal_center = self.success_evaluation(last_position, traj)

            elif self.args.dataset == 'gibson':
                last_x = -last_position[2]
                last_y = -last_position[0]
                last_axis = quaternion.as_euler_angles(last_rotation)[0]
                if (last_axis % (2 * np.pi)) < 0.1 or (last_axis % (2 * np.pi)) > 2 * np.pi - 0.1:
                    o = quaternion.as_euler_angles(last_rotation)[1]
                else:
                    o = 2 * np.pi - quaternion.as_euler_angles(last_rotation)[1]
                if o > np.pi:
                    o -= 2 * np.pi
                min_x, min_y = self.dataset_info[traj['floor_id']]['origin'] / 100.0
                last_x, last_y = int((-last_x - min_x) * 20.), int((-last_y - min_y) * 20.)
                o = np.rad2deg(o) + 180.0
                sim_loc = (last_y, last_x, o)
                min_dist = self.gt_planner.fmm_dist[sim_loc[0], sim_loc[1]] / 20.0
                if min_dist == 0.0:
                    success = 1
                else:
                    success = 0

                spl = min(success * shortest_path_length / self.path_length, 1)






            total_success.append(success)
            total_spl.append(spl)
            if shortest_path_length < 5.0:
                easy_success.append(success)
                easy_spl.append(spl)
                path_level = 'easy'
            elif shortest_path_length < 10.0:
                medium_success.append(success)
                medium_spl.append(spl)
                path_level = 'medium'
            else:
                hard_success.append(success)
                hard_spl.append(spl)
                path_level = 'hard'

            obj_success_list[self.goal_obj_names[self.goal_class_idx]].append(success)
            obj_spl_list[self.goal_obj_names[self.goal_class_idx]].append(spl)

            result = {
                'goal object': self.goal_info['category'],
                'success': success,
                'spl': spl,
                'min_dist_to_viewpoint': float(min_dist_to_viewpoint),
                'min_dist_to_goal_center': float(min_dist_to_goal_center),
                'action step': self.action_step,
                'shortest_path_length': shortest_path_length,
                'path_length': self.path_length,
                'path_level': path_level,
            }


            ## --- save results --- ##
            success_fail = 'success' if success else 'fail'
            save_dir = f'{self.args.save_dir}/{self.data_type}/{self.env_name}/{success_fail}/{self.env_name}_{data_idx:04d}'

            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            self.save_rgbd_video(self.rgb_list, self.depth_list, save_dir)
            # self.save_semantic_video(self.semantic_list, self.args.save_dir, self.env_name, data_idx)
            #
            # self.save_obj_data(obj_data, self.args.save_dir, self.env_name, data_idx, tot=True)

            with open(f'{save_dir}/graph.pkl', 'wb') as f:
                pickle.dump(self.graph_map, f)

            cur_goal_obj_category_name = self.goal_info['category']
            if self.vis_floorplan:


                # vis_graph_map = self.vis_topdown_map_with_captions(self.graph_map,
                #                                                     curr_node=self.cur_node,
                #                                                     curr_position=agent.get_state().position - self.abs_init_position,
                #                                                     visited_positions=self.visited_positions)

                self.save_video(self.vis_traj, save_dir)
                vis_save_dir = [save_dir + '/result.png']
                vis_save_dir.append(
                    f'{self.args.save_dir}/{self.data_type}/{success_fail}/{self.env_name}_{data_idx:04d}_{self.goal_obj_names[self.goal_class_idx]}_{path_level}.png')
                if not os.path.exists(f'{self.args.save_dir}/{self.data_type}/{success_fail}'):
                    os.makedirs(f'{self.args.save_dir}/{self.data_type}/{success_fail}')
                # vis_save_dir = f'{self.args.save_dir}/{self.data_type}/{self.env_name}/{success_fail}/{self.env_name}_{data_idx:04d}/result.png'
                self.save_viewpoint_on_topdown_map(save_dir=vis_save_dir,
                                                   bias_position=self.abs_init_position,
                                                   curr_position=agent.get_state().position - self.abs_init_position,
                                                   curr_goal_position=self.vis_object_goal_position,
                                                   result=result)



            with open(
                    f'{save_dir}/result.json', 'w') as f:
                json.dump(result, f)

            print(
                f"[{env_idx}/{tot_env_num}] {self.env_name} - [{data_idx}/{len(valid_traj_list)}], Time : {time.time() - src_start_time} \n"
                f"         Total - success: {np.mean(total_success)}, spl: {np.mean(total_spl)}, count: {len(total_success)} \n"
                f"         Easy - success: {np.mean(easy_success)}, spl: {np.mean(easy_spl)}, count: {len(easy_success)} \n"
                f"         Medium - success: {np.mean(medium_success)}, spl: {np.mean(medium_spl)}, count: {len(medium_success)} \n"
                f"         Hard - success: {np.mean(hard_success)}, spl: {np.mean(hard_spl)}, count: {len(hard_success)} \n")



        print(f"[{env_idx}/{tot_env_num}] {self.env_name}  Done,   Time : {time.time()-env_start_time}")

        # data_dir = f"{self.args.save_dir}/{self.data_type}/{self.env_name}"
        # if not os.path.exists(data_dir):
        #     err_data_dir = f"{self.args.save_dir}/err/{self.env_name}"
        #     os.makedirs(err_data_dir)


        self._sim.close()
        del self._sim

        if len(total_success) > 0:
            success_results['total']['success'] = np.mean(total_success)
            success_results['total']['spl'] = np.mean(total_spl)
            success_results['total']['count'] = len(total_success)
        if len(easy_success) > 0:
            success_results['easy']['success'] = np.mean(easy_success)
            success_results['easy']['spl'] = np.mean(easy_spl)
            success_results['easy']['count'] = len(easy_success)
        if len(medium_success) > 0:
            success_results['medium']['success'] = np.mean(medium_success)
            success_results['medium']['spl'] = np.mean(medium_spl)
            success_results['medium']['count'] = len(medium_success)
        if len(hard_success) > 0:
            success_results['hard']['success'] = np.mean(hard_success)
            success_results['hard']['spl'] = np.mean(hard_spl)
            success_results['hard']['count'] = len(hard_success)

        for obj_name in obj_success_list.keys():
            if len(obj_success_list[obj_name]) > 0:
                obj_success = np.mean(obj_success_list[obj_name])
                obj_spl = np.mean(obj_spl_list[obj_name])
                obj_count = len(obj_success_list[obj_name])
                obj_success_results[obj_name] = {'success': obj_success, 'spl': obj_spl, 'count': obj_count}


        return success_results, obj_success_results
