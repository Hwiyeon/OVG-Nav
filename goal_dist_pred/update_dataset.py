import os
os.environ["OMP_NUM_THREADS"] = '1'
import sys
sys.path.append('/home/hwing/Projects/OVG-Nav')

import argparse


parser = argparse.ArgumentParser("Pytorch code for unsupervised video summarization with REINFORCE")


parser.add_argument("--data_split", type=int, default=9)
parser.add_argument("--data_split_max", type=int, default=10)

parser.add_argument('--vis_feat_dim', default=512, type=int)
parser.add_argument('--goal_type_num', default=6, type=int)
parser.add_argument('--max_dist', default=5., type=float)
parser.add_argument('--use_cm_score', default=True, type=bool)

# Optimization options
parser.add_argument('--batch-size', type=int, default=32, help="learning rate (default: 1e-05)")
parser.add_argument('--lr', type=float, default=0.1, help="learning rate (default: 1e-05)")
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('--max-epoch', type=int, default=20, help="maximum epoch for training (default: 60)")
parser.add_argument('--stepsize', type=int, default=30, help="how many steps to decay learning rate (default: 30)")
parser.add_argument('--gamma', type=float, default=0.1, help="learning rate decay (default: 0.1)")
parser.add_argument('--beta', type=float, default=0.01, help="weight for summary length penalty term (default: 0.01)")

# Misc
parser.add_argument('--seed', type=int, default=1, help="random seed (default: 1)")
parser.add_argument('--gpu', type=str, default='9', help="which gpu devices to use")
parser.add_argument('--resume', type=str, default='', help="path to resume file")
parser.add_argument('--save-results', action='store_true', help="whether to save  output results")
parser.add_argument('--data-dir', default='/disk4/hwing/Dataset/cm_graph/mp3d/0711/21cat_relative_pose_step_by_step_front_edge1.0', type=str)
parser.add_argument('--log_dir', default='logs/cm_0607/0607_{}_lr{}_test', type=str)
parser.add_argument('--proj_name', default='object_value_graph_estimation', type=str)
parser.add_argument('--disp_iter', type=int, default=10, help="random seed (default: 1)")
parser.add_argument('--save_iter', type=int, default=3, help="random seed (default: 1)")
parser.add_argument('--checkpoints', type=str, default=None)

# --- code test ---
parser.add_argument('--one_iter_test', default=False, type=bool)


args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

from model_value_graph_0607 import TopoGCN_v2 as Model

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

import numpy as np
import random
import time

import pickle
import sys
import torch.optim as optim
import wandb
import cv2
from tqdm import tqdm

from dataloader_batch_graph_data_0607 import Batch_traj_DataLoader_pano_goalscore as Batch_traj_DataLoader
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from utils.obj_category_info import assign_room_category, obj_names_det, mp3d_goal_obj_names, room_names


if args.use_cm_score:
    args.log_dir = args.log_dir.format('use_cm', args.lr)
else:
    args.log_dir = args.log_dir.format('no_cm', args.lr)

print(args)
torch.manual_seed(args.seed)


if not os.path.exists(args.log_dir):
    os.makedirs(args.log_dir)


def main_0():
    use_gpu = torch.cuda.is_available()
    train_envs = [os.path.join(args.data_dir, 'train', name) for name in
                  os.listdir(os.path.join(args.data_dir, 'train'))]
    val_envs = [os.path.join(args.data_dir, 'val', name) for name in os.listdir(os.path.join(args.data_dir, 'val'))]
    train_envs.sort()
    val_envs.sort()


    train_list = []
    for i, env in enumerate(train_envs):
        train_list = train_list + [os.path.join(env, x) for x in os.listdir(env)]
    train_list.sort()
    if args.data_split == args.data_split_max-1:
        train_list = train_list[int(args.data_split * len(train_list) / args.data_split_max):]
    else:
        train_list = train_list[int(args.data_split*len(train_list)/args.data_split_max):int((args.data_split+1)*len(train_list)/args.data_split_max)]



    train_batch_num = int(len(train_list) / args.batch_size)
    train_num = train_batch_num * args.batch_size
    train_dataset = Batch_traj_DataLoader(args, train_list[:train_num])

    val_list = []
    for env in val_envs:
        val_list = val_list + [os.path.join(env, x) for x in os.listdir(env)]
    val_list.sort()
    val_batch_num = int(len(val_list) / args.batch_size)
    val_num = val_batch_num * args.batch_size
    val_dataset = Batch_traj_DataLoader(args, val_list)


    # ## -- add goal text clip feat -- ##
    # with open('/data1/hwing/Projects/offline_objgoal/modules/comet_relation/gibson_goal_obj_text_clip_feat.pkl',
    #           'rb') as f:
    #     goal_text_clip_feat = pickle.load(f)
    # goal_text_clip_feat = goal_text_clip_feat.cpu()

    # ## -- check invalid data -- ##
    invalid_list = []
    # for data in tqdm(val_list, total=len(val_list)):
    #     try:
    #         with open(f'{data}/graph.pkl', 'rb') as f:
    #             graph_data = pickle.load(f)
    #
    #         ## pre compute pano scores
    #         nodes = [graph_data.node_by_id[id] for id in graph_data.node_by_id.keys()]
    #
    #         cand_weight = torch.Tensor(graph_data.goal_cm_info['cand_category_room_score'][:5])
    #
    #         for i in range(len(nodes)):
    #             nodeid = nodes[i].nodeid
    #
    #             node_goal_cm_scores = torch.Tensor(nodes[i].goal_cm_info['goal_cm_scores']) * 0.01
    #             # node_goal_cm_scores = torch.softmax(node_goal_cm_scores, dim=1)
    #             node_goal_cm_scores[torch.isnan(node_goal_cm_scores)] = 0.0  ## maskout nan
    #             graph_data.node_by_id[nodeid].goal_cm_scores = node_goal_cm_scores
    #
    #             node_cand_cm_scores = torch.Tensor(nodes[i].goal_cm_info['cand_cm_scores'][:, :5]) * 0.01
    #             # weighted_cand_cm_scores = torch.softmax(node_cand_cm_scores,
    #             #                                         dim=1) * cand_weight  ## weighted by room category
    #             node_cand_cm_scores[torch.isnan(node_cand_cm_scores)] = 0.0  ## maskout nan
    #             graph_data.node_by_id[nodeid].cand_cm_scores = node_cand_cm_scores
    #
    #             node_pano_vis_feat = nodes[i].clip_feat
    #             node_pano_vis_feat[torch.isnan(node_pano_vis_feat)] = 0.0  ## maskout nan
    #             graph_data.node_by_id[nodeid].pano_vis_feat = node_pano_vis_feat
    #
    #
    #             # node_cand_cm_scores = torch.Tensor(nodes[i].goal_cm_info['cand_cm_scores'][:, :5])
    #             # weighted_cand_cm_scores = torch.softmax(node_cand_cm_scores,
    #             #                                         dim=1) * cand_weight  ## weighted by room category
    #             # weighted_cand_cm_scores[torch.isnan(weighted_cand_cm_scores)] = 0.0  ## maskout nan
    #             # graph_data.node_by_id[nodeid].weighted_cand_cm_scores = weighted_cand_cm_scores
    #             #
    #             #
    #
    #
    #
    #         # goal_class_idx = torch.where(graph_data.node_by_id['0'].goal_cat == 1)[0]
    #         # if len(graph_data.goal_text_clip_feat) == 6:
    #         #     graph_data.goal_text_clip_feat = graph_data.goal_text_clip_feat[goal_class_idx]
    #         # goal_feat = goal_text_clip_feat[goal_class_idx]
    #         # graph_data.goal_class_idx = goal_class_idx
    #         # graph_data.goal_text_clip_feat = goal_feat
    #         # for idx in graph_data.node_by_id.keys():
    #         #     graph_data.node_by_id[idx].goal_cm_info['goal_place_feat'] = graph_data.node_by_id[idx].goal_cm_info[
    #         #         'goal_place_feat'].cpu()
    #
    #         # adj_mtx_vec = np.zeros([len(graph_data.nodes), len(graph_data.nodes), 3])
    #         # for edge in graph_data.edge_ids:
    #         #     p0 = np.array(graph_data.node_by_id[edge[0]].pos)
    #         #     p1 = np.array(graph_data.node_by_id[edge[1]].pos)
    #         #     adj_mtx_vec[int(edge[0]), int(edge[1])] = p1 - p0
    #         # graph_data.adj_mtx_vec = adj_mtx_vec
    #         with open(f'{data}/graph.pkl', 'wb') as f:
    #             pickle.dump(graph_data, f)
    #
    #     except KeyboardInterrupt:
    #         print("KeyboardInterrupt")
    #         break
    #     except:
    #         print(data)
    #         invalid_list.append(data)

    for data in tqdm(train_list, total=len(train_list)):
        try:
            with open(f'{data}/graph.pkl', 'rb') as f:
                graph_data = pickle.load(f)

            ## pre compute pano scores
            nodes = [graph_data.node_by_id[id] for id in graph_data.node_by_id.keys()]

            cand_weight = torch.Tensor(graph_data.goal_cm_info['cand_category_room_score'][:5])

            max_goal_val, max_cand_val = -1, -1
            min_goal_val, min_cand_val = 1, 1

            for i in range(len(nodes)):
                nodeid = nodes[i].nodeid

                node_goal_cm_scores = torch.Tensor(nodes[i].goal_cm_info['goal_cm_scores']) * 0.01
                # node_goal_cm_scores = torch.softmax(node_goal_cm_scores, dim=1)
                node_goal_cm_scores[torch.isnan(node_goal_cm_scores)] = 0.0  ## maskout nan
                graph_data.node_by_id[nodeid].goal_cm_scores = node_goal_cm_scores

                node_cand_cm_scores = torch.Tensor(nodes[i].goal_cm_info['cand_cm_scores'][:, :5]) * 0.01
                # weighted_cand_cm_scores = torch.softmax(node_cand_cm_scores,
                #                                         dim=1) * cand_weight  ## weighted by room category
                node_cand_cm_scores[torch.isnan(node_cand_cm_scores)] = 0.0  ## maskout nan
                graph_data.node_by_id[nodeid].cand_cm_scores = node_cand_cm_scores


                if max_goal_val < torch.max(node_goal_cm_scores):
                    max_goal_val = torch.max(node_goal_cm_scores)
                if min_goal_val > torch.min(node_goal_cm_scores) and torch.min(node_goal_cm_scores) != 0.0:
                    min_goal_val = torch.min(node_goal_cm_scores)
                if max_cand_val < torch.max(node_cand_cm_scores):
                    max_cand_val = torch.max(node_cand_cm_scores)
                if min_cand_val > torch.min(node_cand_cm_scores) and torch.min(node_cand_cm_scores) != 0.0:
                    min_cand_val = torch.min(node_cand_cm_scores)

                node_pano_vis_feat = nodes[i].clip_feat
                node_pano_vis_feat[torch.isnan(node_pano_vis_feat)] = 0.0  ## maskout nan
                graph_data.node_by_id[nodeid].pano_vis_feat = node_pano_vis_feat


                # node_cand_cm_scores = torch.Tensor(nodes[i].goal_cm_info['cand_cm_scores'][:, :5])
                # weighted_cand_cm_scores = torch.softmax(node_cand_cm_scores,
                #                                         dim=1) * cand_weight  ## weighted by room category
                # weighted_cand_cm_scores[torch.isnan(weighted_cand_cm_scores)] = 0.0  ## maskout nan
                # graph_data.node_by_id[nodeid].weighted_cand_cm_scores = weighted_cand_cm_scores
                #



            # goal_class_idx = torch.where(graph_data.node_by_id['0'].goal_cat == 1)[0]
            # if len(graph_data.goal_text_clip_feat) == 6:
            #     graph_data.goal_text_clip_feat = graph_data.goal_text_clip_feat[goal_class_idx]
            # goal_feat = goal_text_clip_feat[goal_class_idx]
            # graph_data.goal_class_idx = goal_class_idx
            # graph_data.goal_text_clip_feat = goal_feat
            # # for idx in graph_data.node_by_id.keys():
            #     graph_data.node_by_id[idx].goal_cm_info['goal_place_feat'] = graph_data.node_by_id[idx].goal_cm_info[
            #         'goal_place_feat'].cpu()

            # adj_mtx_vec = np.zeros([len(graph_data.nodes), len(graph_data.nodes), 3])
            # for edge in graph_data.edge_ids:
            #     p0 = np.array(graph_data.node_by_id[edge[0]].pos)
            #     p1 = np.array(graph_data.node_by_id[edge[1]].pos)
            #     adj_mtx_vec[int(edge[0]), int(edge[1])] = p1 - p0
            # graph_data.adj_mtx_vec = adj_mtx_vec
            #
            with open(f'{data}/graph.pkl', 'wb') as f:
                pickle.dump(graph_data, f)
        except KeyboardInterrupt:
            print("KeyboardInterrupt")
            break
        except:
            print(data)
            # os.system(f"rm -r {data}")
            invalid_list.append(data)

def main():
    use_gpu = torch.cuda.is_available()
    train_envs = [os.path.join(args.data_dir, 'train', name) for name in
                  os.listdir(os.path.join(args.data_dir, 'train'))]
    val_envs = [os.path.join(args.data_dir, 'val', name) for name in os.listdir(os.path.join(args.data_dir, 'val'))]
    train_envs.sort()
    val_envs.sort()


    train_list = []
    for i, env in enumerate(train_envs):
        train_list = train_list + [os.path.join(env, x) for x in os.listdir(env)]
    train_list.sort()
    if args.data_split == args.data_split_max-1:
        train_list = train_list[int(args.data_split * len(train_list) / args.data_split_max):]
    else:
        train_list = train_list[int(args.data_split*len(train_list)/args.data_split_max):int((args.data_split+1)*len(train_list)/args.data_split_max)]



    train_batch_num = int(len(train_list) / args.batch_size)
    train_num = train_batch_num * args.batch_size
    train_dataset = Batch_traj_DataLoader(args, train_list[:train_num])

    val_list = []
    for env in val_envs:
        val_list = val_list + [os.path.join(env, x) for x in os.listdir(env)]
    val_list.sort()
    val_batch_num = int(len(val_list) / args.batch_size)
    val_num = val_batch_num * args.batch_size
    val_dataset = Batch_traj_DataLoader(args, val_list)


    # ## -- add goal text clip feat -- ##
    # with open('/data1/hwing/Projects/offline_objgoal/modules/comet_relation/gibson_goal_obj_text_clip_feat.pkl',
    #           'rb') as f:
    #     goal_text_clip_feat = pickle.load(f)
    # goal_text_clip_feat = goal_text_clip_feat.cpu()

    # ## -- check invalid data -- ##
    invalid_list = []
    for data in tqdm(val_list, total=len(val_list)):
        try:

            result = val_dataset.load_data(data)


        except KeyboardInterrupt:
            print("KeyboardInterrupt")
            break
        except:
            print(data)
            # os.system(f"rm -r {data}")
            invalid_list.append(data)

    # for data in tqdm(train_list, total=len(train_list)):
    #     try:
    #         result = train_dataset.load_data(data)
    #
    #     except KeyboardInterrupt:
    #         print("KeyboardInterrupt")
    #         break
    #     except:
    #         print(data)
    #         os.system(f"rm -r {data}")
    #         invalid_list.append(data)

if __name__ == '__main__':
    main_0()