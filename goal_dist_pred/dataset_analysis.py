import os
os.environ["OMP_NUM_THREADS"] = '1'
import sys
sys.path.append('/home/hwing/Projects/OVG-Nav')

import argparse


parser = argparse.ArgumentParser("Pytorch code for unsupervised video summarization with REINFORCE")

parser.add_argument("--run_type", type=str, default="train")
parser.add_argument("--data_split", type=int, default=0)
parser.add_argument("--data_split_max", type=int, default=1)

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
parser.add_argument('--data-dir', default='/disk4/hwing/Dataset/cm_graph/mp3d/0729/21cat_relative_pose_step_by_step_pano_connect_edge2', type=str)
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
import networkx as nx

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

    path_groups = {}
    for path in train_list:
        # Extract the component part (e.g., 'A/B', 'A/C')
        component = "_".join(path.rsplit('_', 1)[:-1])

        # Extract the suffix and convert it to an integer
        suffix = int(path.rsplit('_', 1)[-1])

        # Check if the component already exists in the dictionary
        if component in path_groups:
            # If the current suffix is greater than the existing maximum, update it
            if suffix > path_groups[component]:
                path_groups[component] = suffix
        else:
            # If the component is not in the dictionary, add it with the current suffix
            path_groups[component] = suffix

    train_end_epi_list = [f"{component}_{suffix:03d}" for component, suffix in path_groups.items()]

    path_groups = {}
    for path in val_list:
        # Extract the component part (e.g., 'A/B', 'A/C')
        component = "_".join(path.rsplit('_', 1)[:-1])

        # Extract the suffix and convert it to an integer
        suffix = int(path.rsplit('_', 1)[-1])

        # Check if the component already exists in the dictionary
        if component in path_groups:
            # If the current suffix is greater than the existing maximum, update it
            if suffix > path_groups[component]:
                path_groups[component] = suffix
        else:
            # If the component is not in the dictionary, add it with the current suffix
            path_groups[component] = suffix

    val_end_epi_list = [f"{component}_{suffix:03d}" for component, suffix in path_groups.items()]



    # ## -- check invalid data -- ##
    invalid_list = []
    if args.run_type == 'val':

        val_graph_mean_length = []
        val_graph_median_length = []
        val_graph_max_length = []
        val_data_cnt = 0


        for data in tqdm(val_end_epi_list, total=len(val_end_epi_list)):
            try:
                with open(f'{data}/graph.pkl', 'rb') as f:
                    graph_data = pickle.load(f)

                G = nx.from_numpy_matrix(graph_data.connection_adj_mtx)
                lengths = nx.floyd_warshall_numpy(G)

                val_graph_mean_length.append(np.mean(lengths))
                val_graph_median_length.append(np.median(np.array(lengths)))
                val_graph_max_length.append(np.max(lengths))
                val_data_cnt += 1

                nodes = [graph_data.node_by_id[id] for id in graph_data.node_by_id.keys()]
                dist = []
                for node in nodes:
                    dist.append(min(node.dist_to_objs))

                dist_diff = []
                score_diff = []
                for i, j in graph_data.edge_ids:
                    dist_diff.append(
                        np.min(graph_data.node_by_id[i].dist_to_objs) - np.min(graph_data.node_by_id[j].dist_to_objs))
                    score_src = max(1 - np.min(graph_data.node_by_id[i].dist_to_objs) / 30, 0)
                    score_tgt = max(1 - np.min(graph_data.node_by_id[j].dist_to_objs) / 30, 0)
                    score_diff.append(score_src - score_tgt)

            except KeyboardInterrupt:
                print("KeyboardInterrupt")
                break
            except:
                print(data)
                invalid_list.append(data)

        print(f'val_graph_mean_length: {np.mean(val_graph_mean_length)}')
        print(f'val_graph_median_length: {np.mean(val_graph_median_length)}')
        print(f'val_graph_max_length: {np.mean(val_graph_max_length)}')
        print(f'val_data_cnt: {val_data_cnt}')

    elif args.run_type == 'train':

        train_graph_mean_length = []
        train_graph_median_length = []
        train_graph_max_length = []
        train_data_cnt = 0

        # for data in tqdm(train_list, total=len(train_list)):
        for data in train_end_epi_list:
            try:
                with open(f'{data}/graph.pkl', 'rb') as f:
                    graph_data = pickle.load(f)

                G = nx.from_numpy_matrix(graph_data.connection_adj_mtx)
                lengths = nx.floyd_warshall_numpy(G)

                train_graph_mean_length.append(np.mean(lengths))
                train_graph_median_length.append(np.median(np.array(lengths)))
                train_graph_max_length.append(np.max(lengths))
                train_data_cnt += 1
                print(f'Data cnt [{train_data_cnt}/{len(train_end_epi_list)}] graph mean length: {np.mean(lengths):.5f}, '
                      f'graph median length: {np.median(np.array(lengths)):.5f}, graph max length: {np.max(lengths):.5f}')

            except KeyboardInterrupt:
                print("KeyboardInterrupt")
                break
            except:
                print(data)
                invalid_list.append(data)

        print(f'train_graph_mean_length: {np.mean(train_graph_mean_length)}')
        print(f'train_graph_median_length: {np.mean(train_graph_median_length)}')
        print(f'train_graph_max_length: {np.mean(train_graph_max_length)}')
        print(f'train_data_cnt: {train_data_cnt}')

if __name__ == '__main__':
    main()