from torch.utils.data import Dataset
from torchvision import transforms
import torch
import cv2
from PIL import Image
import random
import numpy as np
# from torch_geometric.data import Data
import os
import matplotlib.pyplot as plt
from utils.obj_category_info import assign_room_category, obj_names_det, mp3d_goal_obj_names, room_names
import pickle
import json

class Batch_traj_DataLoader():
    def __init__(self, args, data_list):
        self.args = args
        self.data_list = data_list
        self.vis_feat_size = args.vis_feat_dim
        self.feat_size = self.vis_feat_size + 1 + 3 + args.goal_type_num
        self.max_dist = args.max_dist
        self.use_cm_score = args.use_cm_score
        if args.use_cm_score:
            self.feat_size += 1
        self.len_data = len(self.data_list)
        # self.goal_obj_names = goal_obj_names


        self.datasets = {}
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])


    def __getitem__(self, index):

        data_dict = self.load_data(self.data_list[index])
        return data_dict

    def __len__(self):
        return len(self.data_list)


    def load_data(self, data_path):
        """
        :param args:
            --> 1 = forward, 2 = rot_left, 3 = rot_right
        :return: [demon_rgb, demon_action, trial_rgb, trial_action, action_mask]
        """
        ## load seq_data
        data_name = data_path.split('/')[-1]
        # with open(f'{data_path}/result.json', 'r') as f:
        #     result_data = json.load(f)
        with open(f'{data_path}/graph.pkl', 'rb') as f:
            graph_data = pickle.load(f)

        # nodes = list(graph_data.nodes)
        nodes = [graph_data.node_by_id[id] for id in graph_data.node_by_id.keys()]
        edges = list(graph_data.edges)
        graph_size = len(nodes)

        node_goal_dists = torch.zeros([graph_size, 1], dtype=torch.float)
        node_cm_scores = torch.zeros([graph_size, 1], dtype=torch.float)
        node_features = torch.zeros([graph_size, self.feat_size], dtype=torch.float)

        for i in range(graph_size):
            goal_idx = torch.where(nodes[i].goal_cat == 1)[0]
            node_goal_dists[i] = max(1- nodes[i].dist_to_objs[goal_idx] / self.max_dist, 0)
            node_cm_scores[i] = torch.Tensor([nodes[i].cm_score])

        softmax_node_cm_scores = torch.softmax(node_cm_scores, dim=0)

        for i in range(graph_size):
            if self.use_cm_score:
                node_feat = torch.cat([nodes[i].vis_feat,
                                       softmax_node_cm_scores[i],
                                       nodes[i].visited,
                                       torch.Tensor(nodes[i].pos),
                                       nodes[i].goal_cat], dim=0)
            else:
                node_feat = torch.cat([nodes[i].vis_feat,
                                       nodes[i].visited,
                                       torch.Tensor(nodes[i].pos),
                                       nodes[i].goal_cat], dim=0)

            node_features[i] = node_feat

        adj_mtx = torch.zeros([graph_size, graph_size], dtype=torch.float)
        for edge in edges:
            if adj_mtx[int(edge.ids[0]), int(edge.ids[1])] == 0:
                adj_mtx[int(edge.ids[0]), int(edge.ids[1])] = edge.weight.astype(float)
            if adj_mtx[int(edge.ids[1]), int(edge.ids[0])] == 0 and edge.nodes[1].visited:
                adj_mtx[int(edge.ids[1]), int(edge.ids[0])] = edge.weight.astype(float)


        return {
            'node_features': node_features,
            'adj_mtx': adj_mtx,
            'node_goal_dists': node_goal_dists,
        }




class Batch_traj_DataLoader_pano():
    def __init__(self, args, data_list):
        self.args = args
        self.data_list = data_list
        self.vis_feat_size = args.vis_feat_dim
        self.info_dim = 1 + 3  # visited, position
        self.use_cm_score = args.use_cm_score
        if args.use_cm_score:
            self.info_dim += 12 * 5  # cm_score
        self.feat_dim = args.vis_feat_dim + self.info_dim
        self.max_dist = args.max_dist

        self.len_data = len(self.data_list)
        # self.goal_obj_names = goal_obj_names


        self.datasets = {}
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])


    def __getitem__(self, index):

        data_dict = self.load_data(self.data_list[index])
        return data_dict

    def __len__(self):
        return len(self.data_list)


    def load_data(self, data_path):
        """
        :param args:
            --> 1 = forward, 2 = rot_left, 3 = rot_right
        :return: [demon_rgb, demon_action, trial_rgb, trial_action, action_mask]
        """
        ## load seq_data
        data_name = data_path.split('/')[-1]
        # with open(f'{data_path}/result.json', 'r') as f:
        #     result_data = json.load(f)
        with open(f'{data_path}/graph.pkl', 'rb') as f:
            graph_data = pickle.load(f)

        # nodes = list(graph_data.nodes)
        nodes = [graph_data.node_by_id[id] for id in graph_data.node_by_id.keys()]
        edges = list(graph_data.edges)
        graph_size = len(nodes)

        node_goal_dists = torch.zeros([graph_size, 1], dtype=torch.float)
        node_cm_scores = torch.zeros([graph_size, 12 * 5], dtype=torch.float)
        node_features = torch.zeros([graph_size, 12 * self.vis_feat_size], dtype=torch.float)
        node_goal_features = torch.zeros([graph_size, self.vis_feat_size], dtype=torch.float)
        node_info_features = torch.zeros([graph_size, self.info_dim], dtype=torch.float)
        node_pose = torch.zeros([graph_size, 3], dtype=torch.float)
        node_goal_idx = torch.zeros([graph_size, 1], dtype=torch.float)

        cand_weight = torch.Tensor(graph_data.goal_cm_info['cand_category_room_score'][:5])

        for i in range(graph_size):
            goal_idx = torch.where(nodes[i].goal_cat == 1)[0]
            node_goal_idx[i] = goal_idx
            node_goal_dists[i] = max(1- nodes[i].dist_to_objs[goal_idx] / self.max_dist, 0)

            # node_cand_cm_scores = torch.Tensor(nodes[i].goal_cm_info['cand_cm_scores'][:, :5])
            # weighted_cand_cm_scores = torch.softmax(node_cand_cm_scores, dim=1) * cand_weight   ## weighted by room category
            # weighted_cand_cm_scores[torch.isnan(weighted_cand_cm_scores)] = 0.0     ## maskout nan
            # node_cm_scores[i] = torch.reshape(weighted_cand_cm_scores, [-1])
            node_cm_scores[i] = torch.reshape(nodes[i].weighted_cand_cm_scores, [-1])

            node_pose[i] = torch.Tensor(nodes[i].pos)

            # node_pano_vis_feat = nodes[i].clip_feat
            # node_pano_vis_feat[torch.isnan(node_pano_vis_feat)] = 0.0   ## maskout nan
            # node_features[i] = torch.reshape(node_pano_vis_feat, [-1])
            node_features[i] = torch.reshape(nodes[i].pano_vis_feat, [-1])

            node_goal_features[i] = graph_data.goal_text_clip_feat
            if self.use_cm_score:
                node_info_features[i] = torch.cat([nodes[i].visited,
                                                   node_pose[i],
                                                   node_cm_scores[i]], dim=0)
            else:
                node_info_features[i] = torch.cat([nodes[i].visited,
                                                   node_pose[i]], dim=0)

        # adj_mtx = torch.zeros([graph_size, graph_size], dtype=torch.float)
        # for edge in edges:
        #     if adj_mtx[int(edge.ids[0]), int(edge.ids[1])] == 0:
        #         adj_mtx[int(edge.ids[0]), int(edge.ids[1])] = edge.weight.astype(float)
        #     if adj_mtx[int(edge.ids[1]), int(edge.ids[0])] == 0 and edge.nodes[1].visited:
        #         adj_mtx[int(edge.ids[1]), int(edge.ids[0])] = edge.weight.astype(float)
        adj_mtx = torch.Tensor(graph_data.adj_mtx) + torch.eye(graph_size)


        return {
            'node_features': node_features,
            'node_goal_features': node_goal_features,
            'node_info_features': node_info_features,
            'adj_mtx': adj_mtx,
            'node_goal_dists': node_goal_dists,
            'node_pose': node_pose,
            'goal_idx': node_goal_idx,
        }



class Batch_traj_DataLoader_pano_goalscore():
    def __init__(self, args, data_list):
        self.args = args
        self.data_list = data_list
        self.vis_feat_size = args.vis_feat_dim
        self.info_dim = 1 + 3  # visited, position
        self.use_cm_score = args.use_cm_score
        self.cm_type = args.cm_type
        if args.cm_type == 'comet':
            self.cm_num = 10
        elif args.cm_type == 'mp3d':
            self.cm_num = 5
        if args.use_cm_score:
            self.info_dim += 12 * self.cm_num  # cm_score
        self.feat_dim = args.vis_feat_dim + self.info_dim
        self.max_dist = args.max_dist

        self.len_data = len(self.data_list)
        # self.goal_obj_names = goal_obj_names


        self.datasets = {}
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])


    def __getitem__(self, index):

        data_dict = self.load_data(self.data_list[index])
        return data_dict

    def __len__(self):
        return len(self.data_list)


    def load_data(self, data_path):
        """
        :param args:
            --> 1 = forward, 2 = rot_left, 3 = rot_right
        :return: [demon_rgb, demon_action, trial_rgb, trial_action, action_mask]
        """
        ## load seq_data
        data_name = data_path.split('/')[-1]
        # with open(f'{data_path}/result.json', 'r') as f:
        #     result_data = json.load(f)
        with open(f'{data_path}/graph.pkl', 'rb') as f:
            graph_data = pickle.load(f)

        # nodes = list(graph_data.nodes)
        nodes = [graph_data.node_by_id[id] for id in graph_data.node_by_id.keys()]
        edges = list(graph_data.edges)
        graph_size = len(nodes)

        node_goal_dists = torch.zeros([graph_size, 1], dtype=torch.float)
        node_cm_scores = torch.zeros([graph_size, 12 * self.cm_num], dtype=torch.float)
        node_features = torch.zeros([graph_size, 12 * self.vis_feat_size], dtype=torch.float)
        node_goal_features = torch.zeros([graph_size, self.vis_feat_size], dtype=torch.float)
        node_info_features = torch.zeros([graph_size, self.info_dim], dtype=torch.float)
        node_pose = torch.zeros([graph_size, 3], dtype=torch.float)
        node_goal_idx = torch.zeros([graph_size, 1], dtype=torch.long)

        cand_weight = torch.Tensor(graph_data.goal_cm_info['cand_category_room_score'][:5])

        for i in range(graph_size):
            goal_idx = torch.where(nodes[i].goal_cat == 1)[0]
            node_goal_idx[i] = goal_idx
            node_goal_dists[i] = max(1- nodes[i].dist_to_objs[goal_idx] / self.max_dist, 0)

            if self.cm_type == 'comet':
                node_cm_scores[i] = torch.reshape(nodes[i].goal_cm_scores, [-1])
            elif self.cm_type == 'mp3d':
                node_cm_scores[i] = torch.reshape(nodes[i].cand_cm_scores, [-1])

            node_pose[i] = torch.Tensor(nodes[i].pos)
            # pano_vis_feat = nodes[i].clip_feat
            # pano_vis_feat[torch.isnan(pano_vis_feat)] = 0.0
            # node_features[i] = torch.reshape(pano_vis_feat, [-1])
            node_features[i] = torch.reshape(nodes[i].pano_vis_feat, [-1])

            node_goal_features[i] = graph_data.goal_text_clip_feat
            if self.use_cm_score:
                node_info_features[i] = torch.cat([nodes[i].visited,
                                                   node_pose[i],
                                                   node_cm_scores[i]], dim=0)
            else:
                node_info_features[i] = torch.cat([nodes[i].visited,
                                                   node_pose[i]], dim=0)

        try:
            # adj_mtx = graph_data.normalized_weighted_adj_mtx
            # adj_mtx = graph_data.normalized_connection_adj_mtx
            adj_mtx = graph_data.weighted_adj_mtx
        except:
            adj_mtx = graph_data.adj_mtx
        # adj_mtx = torch.Tensor(adj_mtx) + torch.eye(graph_size)

        adj_mtx = torch.Tensor(adj_mtx)
        # adj_mtx = adj_mtx / torch.sum(adj_mtx, dim=0).unsqueeze(1)

        return {
            'node_features': node_features,
            'node_goal_features': node_goal_features,
            'node_info_features': node_info_features,
            'adj_mtx': adj_mtx,
            'node_goal_dists': node_goal_dists,
            'node_pose': node_pose,
            'goal_idx': node_goal_idx,
        }


class Batch_traj_DataLoader_rank():
    def __init__(self, args, data_list):
        self.args = args
        self.data_list = data_list
        self.vis_feat_size = args.vis_feat_dim
        self.info_dim = 1 + 3  # visited, position
        self.use_cm_score = args.use_cm_score
        if args.use_cm_score:
            self.info_dim += 1  # cm_score
        self.feat_dim = args.vis_feat_dim + self.info_dim
        self.max_dist = args.max_dist

        self.len_data = len(self.data_list)
        # self.goal_obj_names = goal_obj_names


        self.datasets = {}
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])


    def __getitem__(self, index):

        data_dict = self.load_data(self.data_list[index])
        return data_dict

    def __len__(self):
        return len(self.data_list)


    def load_data(self, data_path):
        """
        :param args:
            --> 1 = forward, 2 = rot_left, 3 = rot_right
        :return: [demon_rgb, demon_action, trial_rgb, trial_action, action_mask]
        """
        ## load seq_data
        data_name = data_path.split('/')[-1]
        # with open(f'{data_path}/result.json', 'r') as f:
        #     result_data = json.load(f)
        with open(f'{data_path}/graph.pkl', 'rb') as f:
            graph_data = pickle.load(f)

        # nodes = list(graph_data.nodes)
        nodes = [graph_data.node_by_id[id] for id in graph_data.node_by_id.keys()]
        edges = list(graph_data.edges)
        graph_size = len(nodes)

        node_goal_dists = torch.zeros([graph_size, 1], dtype=torch.float)
        node_cm_scores = torch.zeros([graph_size, 1], dtype=torch.float)
        node_features = torch.zeros([graph_size, self.vis_feat_size], dtype=torch.float)
        node_goal_features = torch.zeros([graph_size, self.vis_feat_size], dtype=torch.float)
        node_info_features = torch.zeros([graph_size, self.info_dim], dtype=torch.float)

        for i in range(graph_size):
            goal_idx = torch.where(nodes[i].goal_cat == 1)[0]
            # node_goal_dists[i] = max(1- nodes[i].dist_to_objs[goal_idx] / self.max_dist, 0)
            node_goal_dists[i] = nodes[i].dist_to_objs[goal_idx]
            node_cm_scores[i] = torch.Tensor([nodes[i].cm_score])

        softmax_node_goal_dists = torch.softmax(node_goal_dists, dim=0)
        softmax_node_cm_scores = torch.softmax(node_cm_scores, dim=0)

        for i in range(graph_size):
            node_features[i] = nodes[i].vis_feat
            node_goal_features[i] = graph_data.goal_text_clip_feat
            if self.use_cm_score:
                node_info_features[i] = torch.cat([nodes[i].visited,
                                               torch.Tensor(nodes[i].pos),
                                               softmax_node_cm_scores[i]], dim=0)
            else:
                node_info_features[i] = torch.cat([nodes[i].visited,
                                               torch.Tensor(nodes[i].pos)], dim=0)

        # adj_mtx = torch.zeros([graph_size, graph_size], dtype=torch.float)
        # for edge in edges:
        #     if adj_mtx[int(edge.ids[0]), int(edge.ids[1])] == 0:
        #         adj_mtx[int(edge.ids[0]), int(edge.ids[1])] = edge.weight.astype(float)
        #     if adj_mtx[int(edge.ids[1]), int(edge.ids[0])] == 0 and edge.nodes[1].visited:
        #         adj_mtx[int(edge.ids[1]), int(edge.ids[0])] = edge.weight.astype(float)
        adj_mtx = torch.Tensor(graph_data.adj_mtx) + torch.eye(graph_size)


        return {
            'node_features': node_features,
            'node_goal_features': node_goal_features,
            'node_info_features': node_info_features,
            'adj_mtx': adj_mtx,
            'node_goal_dists': softmax_node_goal_dists,
        }



class Batch_traj_DataLoader_pano_goalscore_minmax_diff():
    def __init__(self, args, data_list):
        self.args = args
        self.data_list = data_list
        self.vis_feat_size = args.vis_feat_dim
        self.info_dim = 1 + 3  # visited, position
        self.use_cm_score = args.use_cm_score
        if args.use_cm_score:
            self.info_dim += 12 * 10  # cm_score
        self.feat_dim = args.vis_feat_dim + self.info_dim
        self.max_dist = args.max_dist

        self.len_data = len(self.data_list)
        # self.goal_obj_names = goal_obj_names


        self.datasets = {}
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])


    def __getitem__(self, index):

        data_dict = self.load_data(self.data_list[index])
        return data_dict

    def __len__(self):
        return len(self.data_list)


    def load_data(self, data_path):
        """
        :param args:
            --> 1 = forward, 2 = rot_left, 3 = rot_right
        :return: [demon_rgb, demon_action, trial_rgb, trial_action, action_mask]
        """
        ## load seq_data
        data_name = data_path.split('/')[-1]
        # with open(f'{data_path}/result.json', 'r') as f:
        #     result_data = json.load(f)
        with open(f'{data_path}/graph.pkl', 'rb') as f:
            graph_data = pickle.load(f)

        # nodes = list(graph_data.nodes)
        nodes = [graph_data.node_by_id[id] for id in graph_data.node_by_id.keys()]
        edges = list(graph_data.edges)
        graph_size = len(nodes)

        node_goal_dists = torch.zeros([graph_size, 1], dtype=torch.float)
        node_cm_scores = torch.zeros([graph_size, 12 * 10], dtype=torch.float)
        node_features = torch.zeros([graph_size, 12 * self.vis_feat_size], dtype=torch.float)
        node_goal_features = torch.zeros([graph_size, self.vis_feat_size], dtype=torch.float)
        node_info_features = torch.zeros([graph_size, self.info_dim], dtype=torch.float)
        node_pose = torch.zeros([graph_size, 3], dtype=torch.float)
        node_goal_idx = torch.zeros([graph_size, 1], dtype=torch.long)

        cand_weight = torch.Tensor(graph_data.goal_cm_info['cand_category_room_score'][:5])

        for i in range(graph_size):
            goal_idx = torch.where(nodes[i].goal_cat == 1)[0]
            node_goal_idx[i] = goal_idx
            node_goal_dists[i] = max(1- nodes[i].dist_to_objs[goal_idx] / self.max_dist, 0)

            node_cm_scores[i] = torch.reshape(nodes[i].goal_cm_scores, [-1])

            node_pose[i] = torch.Tensor(nodes[i].pos)
            # pano_vis_feat = nodes[i].clip_feat
            # pano_vis_feat[torch.isnan(pano_vis_feat)] = 0.0
            # node_features[i] = torch.reshape(pano_vis_feat, [-1])
            node_features[i] = torch.reshape(nodes[i].pano_vis_feat, [-1])

            node_goal_features[i] = graph_data.goal_text_clip_feat
            if self.use_cm_score:
                node_info_features[i] = torch.cat([nodes[i].visited,
                                                   node_pose[i],
                                                   node_cm_scores[i]], dim=0)
            else:
                node_info_features[i] = torch.cat([nodes[i].visited,
                                                   node_pose[i]], dim=0)

        try:
            adj_mtx = graph_data.weighted_adj_mtx
            # adj_mtx = graph_data.normalized_connection_adj_mtx
        except:
            adj_mtx = graph_data.adj_mtx
        # adj_mtx = torch.Tensor(adj_mtx) + torch.eye(graph_size)

        adj_mtx = torch.Tensor(adj_mtx)
        # adj_mtx = adj_mtx / torch.sum(adj_mtx, dim=0).unsqueeze(1)

        # min_max_mtx = torch.zeros_like(adj_mtx)
        # min_max_mtx[torch.argmin(node_goal_dists), torch.argmax(node_goal_dists)] = 1
        hop_mtx = torch.Tensor(adj_mtx)
        for i in range(self.args.gcn_layers-1):
            hop_mtx = hop_mtx @ adj_mtx
        min_max_mtx = (hop_mtx > 0).float()

        # visited_nodes = [int(nodeid) for nodeid in graph_data.visited_node_ids]
        # visited_max_mtx = torch.zeros_like(adj_mtx)
        # visited_max_mtx[visited_nodes] = min_max_mtx[visited_nodes]

        return {
            'node_features': node_features,
            'node_goal_features': node_goal_features,
            'node_info_features': node_info_features,
            'adj_mtx': adj_mtx,
            'node_goal_dists': node_goal_dists,
            'node_pose': node_pose,
            'goal_idx': node_goal_idx,
            'min_max_mtx': min_max_mtx,
        }



class Batch_traj_DataLoader_pano_goalscore_nopos():
    def __init__(self, args, data_list):
        self.args = args
        self.data_list = data_list
        self.vis_feat_size = args.vis_feat_dim
        self.info_dim = 1   # visited
        self.use_cm_score = args.use_cm_score
        self.cm_type = args.cm_type
        if args.cm_type == 'comet':
            self.cm_num = 10
        elif args.cm_type == 'mp3d':
            self.cm_num = 5
        if args.use_cm_score:
            self.info_dim += 12 * self.cm_num  # cm_score
        self.feat_dim = args.vis_feat_dim + self.info_dim
        self.max_dist = args.max_dist

        self.len_data = len(self.data_list)
        # self.goal_obj_names = goal_obj_names


        self.datasets = {}
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])


    def __getitem__(self, index):

        data_dict = self.load_data(self.data_list[index])
        return data_dict

    def __len__(self):
        return len(self.data_list)


    def load_data(self, data_path):
        """
        :param args:
            --> 1 = forward, 2 = rot_left, 3 = rot_right
        :return: [demon_rgb, demon_action, trial_rgb, trial_action, action_mask]
        """
        ## load seq_data
        data_name = data_path.split('/')[-1]
        # with open(f'{data_path}/result.json', 'r') as f:
        #     result_data = json.load(f)
        with open(f'{data_path}/graph.pkl', 'rb') as f:
            graph_data = pickle.load(f)

        # nodes = list(graph_data.nodes)
        nodes = [graph_data.node_by_id[id] for id in graph_data.node_by_id.keys()]
        edges = list(graph_data.edges)
        graph_size = len(nodes)

        node_goal_dists = torch.zeros([graph_size, 1], dtype=torch.float)
        node_cm_scores = torch.zeros([graph_size, 12 * self.cm_num], dtype=torch.float)
        node_features = torch.zeros([graph_size, 12 * self.vis_feat_size], dtype=torch.float)
        node_goal_features = torch.zeros([graph_size, self.vis_feat_size], dtype=torch.float)
        node_info_features = torch.zeros([graph_size, self.info_dim], dtype=torch.float)
        node_pose = torch.zeros([graph_size, 3], dtype=torch.float)
        node_goal_idx = torch.zeros([graph_size, 1], dtype=torch.long)

        cand_weight = torch.Tensor(graph_data.goal_cm_info['cand_category_room_score'][:5])

        for i in range(graph_size):
            goal_idx = torch.where(nodes[i].goal_cat == 1)[0]
            node_goal_idx[i] = goal_idx
            node_goal_dists[i] = max(1- nodes[i].dist_to_objs[goal_idx] / self.max_dist, 0)

            if self.cm_type == 'comet':
                node_cm_scores[i] = torch.reshape(nodes[i].goal_cm_scores, [-1])
            elif self.cm_type == 'mp3d':
                node_cm_scores[i] = torch.reshape(nodes[i].cand_cm_scores, [-1])

            node_pose[i] = torch.Tensor(nodes[i].pos)
            node_features[i] = torch.reshape(nodes[i].pano_vis_feat, [-1])

            node_goal_features[i] = graph_data.goal_text_clip_feat
            if self.use_cm_score:
                node_info_features[i] = torch.cat([nodes[i].visited,
                                                   # node_pose[i],
                                                   node_cm_scores[i]], dim=0)
            else:
                node_info_features[i] = nodes[i].visited
                # node_info_features[i] = torch.cat([nodes[i].visited,
                #                                    node_pose[i]], dim=0)

        try:
            # adj_mtx = graph_data.normalized_weighted_adj_mtx
            # adj_mtx = graph_data.normalized_connection_adj_mtx
            adj_mtx = graph_data.weighted_adj_mtx
        except:
            adj_mtx = graph_data.adj_mtx
        # adj_mtx = torch.Tensor(adj_mtx) + torch.eye(graph_size)

        adj_mtx = torch.Tensor(adj_mtx)
        # adj_mtx = adj_mtx / torch.sum(adj_mtx, dim=0).unsqueeze(1)

        return {
            'node_features': node_features,
            'node_goal_features': node_goal_features,
            'node_info_features': node_info_features,
            'adj_mtx': adj_mtx,
            'node_goal_dists': node_goal_dists,
            'node_pose': node_pose,
            'goal_idx': node_goal_idx,
        }

def main():
    return

if __name__ == '__main__':
    main()