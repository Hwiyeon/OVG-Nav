import os
os.environ["OMP_NUM_THREADS"] = '1'
import sys
sys.path.append('/home/hwing/Projects/OVG-Nav')

import argparse


parser = argparse.ArgumentParser("Pytorch code for unsupervised video summarization with REINFORCE")


parser.add_argument('--vis_feat_dim', default=512, type=int)
parser.add_argument('--hidden_size', default=512, type=int)
parser.add_argument('--gcn_layers', default=10, type=int)
# parser.add_argument('--goal_type_num', default=6, type=int)
parser.add_argument('--max_dist', default=30., type=float)
parser.add_argument('--use_cm_score', default=True, type=bool)
parser.add_argument('--goal_cat', type=str, default='mp3d_21')
parser.add_argument('--value_loss_cf', default=1.0, type=float)
parser.add_argument('--adj_loss_cf', default=1.0, type=float)
parser.add_argument('--adj_sim_loss_cf', default=1.0, type=float)
parser.add_argument('--sign_loss_cf', default=0.05, type=float)


# Optimization options
parser.add_argument('--batch-size', type=int, default=512, help="learning rate (default: 1e-05)")
parser.add_argument('--lr', type=float, default=0.001, help="learning rate (default: 1e-05)")
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('--max-epoch', type=int, default=9, help="maximum epoch for training (default: 60)")
parser.add_argument('--stepsize', type=int, default=30, help="how many steps to decay learning rate (default: 30)")
parser.add_argument('--gamma', type=float, default=0.1, help="learning rate decay (default: 0.1)")
parser.add_argument('--beta', type=float, default=0.01, help="weight for summary length penalty term (default: 0.01)")

# Misc
parser.add_argument('--seed', type=int, default=100, help="random seed (default: 1)")
parser.add_argument('--gpu', type=str, default='1', help="which gpu devices to use")
parser.add_argument('--resume', type=str, default='', help="path to resume file")
parser.add_argument('--save-results', action='store_true', help="whether to save  output results")
# parser.add_argument('--data-dir', default='/disk4/hwing/Dataset/cm_graph/mp3d/0630/relative_pose_step_by_step_pano', type=str)
# parser.add_argument('--data-dir', default='/disk4/hwing/Dataset/cm_graph/mp3d/0715/21cat_relative_pose_step_by_step_pano_connect', type=str)
parser.add_argument('--data-dir', default='/disk4/hwing/Dataset/cm_graph/mp3d/0803/21cat_relative_pose_step_by_step_pano_connect_v2', type=str)
# parser.add_argument('--data-dir', default='/disk4/hwing/Dataset/cm_graph/mp3d/0729/21cat_relative_pose_step_by_step_pano_connect_edge1_v2', type=str)
# parser.add_argument('--data-dir_aug', default='/data1/hwing/Dataset/cm_graph/mp3d/0607/random_path_collection_3interval_pure_cmv2', type=str)
# parser.add_argument('--data-dir_aug', default=[
#                                                 '/disk4/hwing/Dataset/cm_graph/mp3d/0803/21cat_relative_pose_step_by_step_pano_connect_v2',
#                                               '/disk4/hwing/Dataset/cm_graph/mp3d/0729/21cat_relative_pose_step_by_step_pano_connect_edge1_v2',
                                               # '/data1/hwing/Dataset/cm_graph/mp3d/0607/random_path_collection_3interval_pure_cm',
                                               # '/home/hwing/Dataset/cm_graph/mp3d/0607/shortest_path_crop_collection_3interval_pure_cm',
                                               # '/home/hwing/Dataset/cm_graph/mp3d/0607/shortest_path_crop_collection_3interval_pure_cm_aug2',
                                               # ])
# parser.add_argument('--data-dir_aug', default=None, type=str)
# parser.add_argument('--data-dir_aug2', default='/home/hwing/Dataset/cm_graph/mp3d/0607/shortest_path_crop_collection_3interval_pure_cm_aug2', type=str)
# parser.add_argument('--data-dir_aug2', default=None, type=str)
parser.add_argument('--log_dir', default='logs/cm_{}/{}_mp3d21_edge1_datav2_panov3_2_layer{}_hidden{}_goalscore_wnorm_adjmtx_valueloss{}_adjloss{}_hoploss_visited{}_signloss{}_{}_maxdist{}_lr{}', type=str)
parser.add_argument('--proj_name', default='object_value_graph_estimation_mp3d21_pano_running_addnode_edge1v1', type=str)
parser.add_argument('--disp_iter', type=int, default=10, help="random seed (default: 1)")
parser.add_argument('--save_iter', type=int, default=3, help="random seed (default: 1)")
parser.add_argument('--checkpoints', type=str, default=None)

# --- code test ---
parser.add_argument('--one_iter_test', default=False, type=bool)


args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

from model_value_graph_0607 import TopoGCN_value_diff_pano_goalscore as Model

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

import numpy as np
import random
import time
from datetime import datetime

import pickle
import sys
import torch.optim as optim
import wandb
import cv2
from tqdm import tqdm

from dataloader_batch_graph_data_0607 import Batch_traj_DataLoader_pano_goalscore_minmax_diff as Batch_traj_DataLoader
# from dataloader_batch_graph_data_0607 import Batch_traj_DataLoader_rank as Batch_traj_DataLoader
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from utils.obj_category_info import assign_room_category, obj_names_det, mp3d_goal_obj_names, room_names, obj_names, gibson_goal_obj_names


cur_date = str(datetime.now().date())
cur_time = str(datetime.now().time())[:5].replace(":", "-")

if args.use_cm_score:
    args.log_dir = args.log_dir.format(cur_date, cur_time, args.gcn_layers, args.hidden_size, args.value_loss_cf, args.adj_loss_cf, args.adj_sim_loss_cf, args.sign_loss_cf, 'use_cm', args.max_dist, args.lr)
else:
    args.log_dir = args.log_dir.format(cur_date, cur_time, args.gcn_layers, args.hidden_size, args.value_loss_cf, args.adj_loss_cf, args.adj_sim_loss_cf, args.sign_loss_cf, 'no_cm', args.max_dist, args.lr)

print(args)
torch.manual_seed(args.seed)


if not os.path.exists(args.log_dir):
    os.makedirs(args.log_dir)


def sign_penalty_loss(y_pred, y_true):
    signs_true = torch.sign(y_true)
    signs_pred = torch.sign(y_pred)

    # compute the loss as the mean squared error between the signs
    loss = torch.mean((signs_true - signs_pred) ** 2)
    return loss


def main():
    use_gpu = torch.cuda.is_available()
    train_envs = [os.path.join(args.data_dir, 'train', name) for name in
                  os.listdir(os.path.join(args.data_dir, 'train'))]
    val_envs = [os.path.join(args.data_dir, 'val', name) for name in os.listdir(os.path.join(args.data_dir, 'val'))]
    train_envs.sort()
    val_envs.sort()

    if args.data_dir_aug is not None:
        for data_aug in args.data_dir_aug:
            train_envs_aug = [os.path.join(data_aug, 'train', name) for name in
                            os.listdir(os.path.join(data_aug, 'train'))]
            val_envs_aug = [os.path.join(data_aug, 'val', name) for name in os.listdir(os.path.join(data_aug, 'val'))]
            train_envs_aug.sort()
            val_envs_aug.sort()
            train_envs = train_envs + train_envs_aug
            val_envs = val_envs + val_envs_aug





    train_list = []
    for i, env in enumerate(train_envs):
        train_list = train_list + [os.path.join(env, x) for x in os.listdir(env)]
    train_list.sort()
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




    if args.goal_cat == 'mp3d':
        goal_obj_names = mp3d_goal_obj_names
    elif args.goal_cat == 'mp3d_21':
        goal_obj_names = obj_names
    elif args.goal_cat == 'gibson':
        goal_obj_names = gibson_goal_obj_names




    def make_collate_batch(samples):
        node_features = torch.cat([sample['node_features'] for sample in samples], dim=0)
        node_info_features = torch.cat([sample['node_info_features'] for sample in samples], dim=0)
        node_goal_features = torch.cat([sample['node_goal_features'] for sample in samples], dim=0)
        node_goal_dists = torch.cat([sample['node_goal_dists'] for sample in samples], dim=0)
        node_pose = torch.cat([sample['node_pose'] for sample in samples], dim=0)
        goal_idx = torch.cat([sample['goal_idx'] for sample in samples], dim=0)
        

        batch_size = node_features.size()[0]
        adj_mtx = torch.zeros(batch_size, batch_size)
        min_max_mtx = torch.zeros(batch_size, batch_size)
        adj_starting_point = 0
        for i, sample in enumerate(samples):
            adj_size = sample['adj_mtx'].size()[0]
            adj_mtx[adj_starting_point:adj_starting_point+adj_size,
                    adj_starting_point:adj_starting_point+adj_size] = sample['adj_mtx']
            min_max_mtx[adj_starting_point:adj_starting_point+adj_size,
                    adj_starting_point:adj_starting_point+adj_size] = sample['min_max_mtx']
            adj_starting_point += adj_size
        return {
            'node_features': node_features,
            'node_info_features': node_info_features,
            'node_goal_features': node_goal_features,
            'adj_mtx': adj_mtx,
            'node_goal_dists': node_goal_dists,
            'node_pose': node_pose,
            'goal_idx': goal_idx,
            'min_max_mtx': min_max_mtx
        }

    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, collate_fn=make_collate_batch, shuffle=True, num_workers=8)
    valid_loader = DataLoader(dataset=val_dataset, batch_size=1, collate_fn=make_collate_batch, num_workers=8)



    if use_gpu:
        print("Currently using GPU {}".format(args.gpu))
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(args.seed)
        dev = "cuda:0"
    else:
        print("Currently using CPU")

    model = Model(args, args.hidden_size)

    print("Model size: {:.5f}M".format(sum(p.numel() for p in model.parameters())/1000000.0))


    if args.resume:
        print("Loading checkpoint from '{}'".format(args.resume))
        model = torch.load(args.resume)
        # model.load_state_dict(checkpoint)
    else:
        start_epoch = 0

    if use_gpu:
        model = nn.DataParallel(model).cuda()


    if not 'test' in args.log_dir: # and not args.one_iter_test:
        wandb.login(key='3e0c4716deb217aacb0f5f3991ac5c30637c8a7a')
        wandb.init(
            project=args.proj_name,
            name=args.log_dir.split('/')[-1],
        )
        wandb.config.update(args)
        wandb.define_metric("epoch")
        wandb.define_metric("val/loss", step_metric="epoch")
        wandb.define_metric("val/value_loss", step_metric="epoch")
        wandb.define_metric("val/adj_loss", step_metric="epoch")
        wandb.define_metric("val/adj_sim_loss", step_metric="epoch")
        wandb.define_metric("val/sign_loss", step_metric="epoch")
        wandb.define_metric("val/value_acc", step_metric="epoch")
        wandb.define_metric("val/rank_acc_3", step_metric="epoch")
        wandb.define_metric("val/rank_acc_1", step_metric="epoch")
        wandb.define_metric("val/rank_score", step_metric="epoch")
        wandb.define_metric("val/pred_diff", step_metric="epoch")
        wandb.define_metric("val/edge_sign_acc", step_metric="epoch")


        for obj_name in goal_obj_names:
            wandb.define_metric("val/obj_value_acc/{}".format(obj_name), step_metric="epoch")
            wandb.define_metric("val/obj_rank_acc_3/{}".format(obj_name), step_metric="epoch")
            wandb.define_metric("val/obj_rank_acc_1/{}".format(obj_name), step_metric="epoch")
            wandb.define_metric("val/obj_rank_score/{}".format(obj_name), step_metric="epoch")
            wandb.define_metric("val/obj_pred_diff/{}".format(obj_name), step_metric="epoch")
            wandb.define_metric("val/edge_sign_acc/{}".format(obj_name), step_metric="epoch")



    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)




    print("==> Start training")



    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.96)
    def lambda_rule(epoch):
        if epoch < int(args.max_epoch /4) * train_batch_num:
            return 1.0
        elif epoch < int(args.max_epoch * 2 / 4) * train_batch_num:
            return 0.1
        elif epoch < int(args.max_epoch * 3 / 4) * train_batch_num:
            return 0.01
        else:
            return 0.001

    # lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    # lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 0.999 ** epoch)
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: (0.1**(1/(train_batch_num*3))) ** epoch)

    mse = nn.MSELoss()
    bce = nn.BCEWithLogitsLoss()
    cross_entropy = nn.CrossEntropyLoss()


    criterion = nn.MSELoss()
    # criterion = nn.KLDivLoss(reduction='batchmean')

    lowest_total_loss = 10000

    iter = 0
    val_cnt = 0
    train_start_time = time.time()

    eyes = torch.eye(args.batch_size * 20).cuda()

    for epoch in range(start_epoch, args.max_epoch):
        model.train()
        start_time = time.time()

        cnt = 0
        disp_loss = 0.0
        disp_value_loss = 0.0
        disp_adj_loss = 0.0
        disp_adj_sim_loss = 0.0
        disp_sign_loss = 0.0
        disp_value_acc = 0.0


        disp_iter = 0

        # for i, data in enumerate(train_loader, 0):
        for data in train_loader:
            disp_iter += 1

            features = data['node_features'].cuda()
            info_features = data['node_info_features'].cuda()
            goal_features = data['node_goal_features'].cuda()
            adj_mtx = data['adj_mtx'].cuda()
            node_goal_dists = data['node_goal_dists'].cuda()
            min_max_mtx = data['min_max_mtx'].cuda()

            pred_dist = model(features, goal_features, info_features, adj_mtx)

            optimizer.zero_grad()
            if args.value_loss_cf == 0:
                value_loss = torch.tensor(0.0).cuda()
            else:
                value_loss = criterion(pred_dist, node_goal_dists)

            ### for adj mtx
            if len(eyes) < len(adj_mtx):
                eyes = torch.eye(len(adj_mtx)).cuda()
            adj_mtx_dig0 = adj_mtx * (1-eyes[:len(adj_mtx), :len(adj_mtx)])
            indices = torch.nonzero(adj_mtx_dig0, as_tuple=True)
            min_max_mtx0 = min_max_mtx * (1-eyes[:len(adj_mtx), :len(adj_mtx)])
            min_max_indices = torch.nonzero(min_max_mtx0, as_tuple=True)

            if args.adj_loss_cf == 0:
                adj_loss = torch.tensor(0.0).cuda()
            else:
                adj_loss = criterion(pred_dist[indices[0]] - pred_dist[indices[1]], node_goal_dists[indices[0]] - node_goal_dists[indices[1]])

            if args.adj_sim_loss_cf == 0:
                adj_sim_loss = torch.tensor(0.0).cuda()
            else:
                adj_sim_loss = criterion(pred_dist[min_max_indices[0]] - pred_dist[min_max_indices[1]], node_goal_dists[min_max_indices[0]] - node_goal_dists[min_max_indices[1]])
                if torch.isnan(adj_loss):
                    adj_loss = torch.tensor(0.0).cuda()

            if args.sign_loss_cf == 0:
                sign_loss = torch.tensor(0.0).cuda()
            else:
                sign_loss = sign_penalty_loss(pred_dist[min_max_indices[0]] - pred_dist[min_max_indices[1]], node_goal_dists[min_max_indices[0]] - node_goal_dists[min_max_indices[1]])

            loss = args.value_loss_cf * value_loss + args.adj_loss_cf * adj_loss + args.sign_loss_cf * sign_loss + args.adj_sim_loss_cf * adj_sim_loss

            loss.backward()
            optimizer.step()

            disp_loss += loss.item()
            disp_value_loss += args.value_loss_cf * value_loss.item()
            disp_adj_loss += args.adj_loss_cf * adj_loss.item()
            disp_adj_sim_loss += args.adj_sim_loss_cf * adj_sim_loss.item()
            disp_sign_loss += args.sign_loss_cf * sign_loss.item()
            disp_value_acc += torch.mean(torch.where(abs(pred_dist-node_goal_dists) <= 0.1 ,1, 0).float())

            cnt += 1
            iter += 1

            lr_scheduler.step()

            if cnt % args.disp_iter == 0:
                end_time = time.time()

                print(f'[Epoch: {epoch+1}/{args.max_epoch}] {cnt}/{train_batch_num} iter, loss: {disp_loss / disp_iter:.5f}, '
                      f'value_loss: {disp_value_loss / disp_iter:.5f}, adj_loss: {disp_adj_loss / disp_iter:.5f}, '
                        f'adj_sim_loss: {disp_adj_sim_loss / disp_iter:.5f}, '
                      f'sign_loss: {disp_sign_loss / disp_iter:.5f}, '
                      f'value_acc: {disp_value_acc / (disp_iter):.5f},'
                      f'lr: {lr_scheduler.get_lr()[0]:.5f}, time: {(end_time - start_time)//60:.0f}:{(end_time - start_time)%60:.0f}'
                      )

                metrics = {
                    'train/loss': float(disp_loss / (disp_iter)),
                    'train/value_loss': float(disp_value_loss / (disp_iter)),
                    'train/adj_loss': float(disp_adj_loss / (disp_iter)),
                    'train/adj_sim_loss': float(disp_adj_sim_loss / (disp_iter)),
                    'train/sign_loss': float(disp_sign_loss / (disp_iter)),
                    'train/acc': float(disp_value_acc / (disp_iter)),
                    'train/lr': lr_scheduler.get_lr()[0],
                }
                if not 'test' in args.log_dir and not args.one_iter_test:
                    wandb.log(metrics)

                disp_loss = 0.0
                disp_value_loss = 0.0
                disp_adj_loss = 0.0
                disp_adj_sim_loss = 0.0
                disp_sign_loss = 0.0
                disp_value_acc = 0.0

                disp_iter = 0

                if args.one_iter_test:
                    break


        val_cnt+=1
        model.eval()
        with torch.no_grad():
            disp_loss = 0.0
            disp_value_loss = 0.0
            disp_adj_loss = 0.0
            disp_adj_sim_loss = 0.0
            disp_sign_loss = 0.0
            disp_value_acc = 0.0
            disp_rank_acc_3 = 0.0
            disp_rank_acc_1 = 0.0
            disp_rank_score = 0.0
            disp_pred_diff = 0.0
            disp_edge_sign_acc = 0.0
            cnt_in_val = 0
            start_time = time.time()
            # for i, data in enumerate(valid_loader, 0):
            obj_results = {}
            for obj_name in goal_obj_names:
                obj_results[obj_name] = {'count': 0, 'disp_value_acc': 0, 'disp_rank_acc_3': 0, 'disp_rank_acc_1': 0,
                                         'disp_rank_score':0, 'disp_pred_diff': 0, 'disp_edge_sign_acc': 0}

            for data in tqdm(valid_loader, total=len(val_list)):

                features = data['node_features'].cuda()
                info_features = data['node_info_features'].cuda()
                goal_features = data['node_goal_features'].cuda()
                adj_mtx = data['adj_mtx'].cuda()
                node_goal_dists = data['node_goal_dists'].cuda()
                goal_idx = data['goal_idx']
                min_max_mtx = data['min_max_mtx'].cuda()

                cand_nodes = 1 - info_features[:, 0]
                if torch.sum(cand_nodes) < 3:
                    continue

                cnt_in_val += 1

                pred_dist = model(features, goal_features, info_features, adj_mtx)

                optimizer.zero_grad()

                value_loss = criterion(pred_dist, node_goal_dists)

                ### for adj mtx
                if len(eyes) < len(adj_mtx):
                    eyes = torch.eye(len(adj_mtx)).cuda()
                adj_mtx_dig0 = adj_mtx * (1 - eyes[:len(adj_mtx), :len(adj_mtx)])
                indices = torch.nonzero(adj_mtx_dig0, as_tuple=True)
                min_max_mtx0 = min_max_mtx * (1 - eyes[:len(adj_mtx), :len(adj_mtx)])
                min_max_indices = torch.nonzero(min_max_mtx0, as_tuple=True)
                
                adj_loss = criterion(pred_dist[indices[0]] - pred_dist[indices[1]],
                                     node_goal_dists[indices[0]] - node_goal_dists[indices[1]])
                if torch.isnan(adj_loss):
                    adj_loss = torch.tensor(0.0).cuda()

                adj_sim_loss = criterion(pred_dist[min_max_indices[0]] - pred_dist[min_max_indices[1]], node_goal_dists[min_max_indices[0]] - node_goal_dists[min_max_indices[1]])
                if torch.isnan(adj_sim_loss):
                    adj_sim_loss = torch.tensor(0.0).cuda()

                sign_loss = sign_penalty_loss(pred_dist[min_max_indices[0]] - pred_dist[min_max_indices[1]],
                                              node_goal_dists[min_max_indices[0]] - node_goal_dists[min_max_indices[1]])

                loss = args.value_loss_cf * value_loss + args.adj_loss_cf * adj_loss + args.sign_loss_cf * sign_loss + args.adj_sim_loss_cf * adj_sim_loss

                disp_loss += loss.item()
                disp_value_loss += args.value_loss_cf * value_loss.item()
                disp_adj_loss += args.adj_loss_cf * adj_loss.item()
                disp_adj_sim_loss += args.adj_sim_loss_cf * adj_sim_loss.item()
                disp_sign_loss += args.sign_loss_cf * sign_loss.item()
                value_acc = torch.mean(torch.where(abs(pred_dist - node_goal_dists) <= 0.1, 1, 0).float())
                # if node_goal_dists.size()[0] >= 3:
                #     topk_list = torch.topk(node_goal_dists, 3, dim=0).indices
                # else:
                #     topk_list = torch.topk(node_goal_dists, node_goal_dists.size()[0], dim=0).indices
                # cand_nodes = 1 - info_features[:, 0]
                if torch.sum(cand_nodes) == 0:
                    topk_list = None
                elif torch.sum(cand_nodes) >= 3:
                    topk_list = torch.topk(node_goal_dists[cand_nodes>0], 3, dim=0).indices
                else:
                    topk_list = torch.topk(node_goal_dists[cand_nodes>0], int(torch.sum(1- info_features[:,0])), dim=0).indices

                if topk_list is None:
                    rank_acc_3 = float(torch.Tensor([1]))
                    rank_acc_1 = float(torch.Tensor([1]))
                else:
                    rank_acc_3 = float(torch.argmax(pred_dist[cand_nodes > 0], dim=0) in topk_list)
                    rank_acc_1 = float(torch.argmax(pred_dist[cand_nodes > 0], dim=0) in topk_list[0])

                oracle_rank = torch.topk(node_goal_dists[cand_nodes > 0], int(torch.sum(cand_nodes)), dim=0).indices[:,0]
                rank_score = 1 - float((torch.where(oracle_rank == torch.argmax(pred_dist[cand_nodes > 0], dim=0))[0]) / torch.sum(cand_nodes))

                pred_diff = np.linalg.norm(
                    data['node_pose'][torch.argmax(pred_dist)] - data['node_pose'][torch.argmax(node_goal_dists)])

                edge_sign_acc = float(torch.mean(((pred_dist[indices[0]] - pred_dist[indices[1]]) *
                                           (node_goal_dists[indices[0]] - node_goal_dists[indices[1]]) > 0
                                           ).float()))

                disp_value_acc += value_acc
                disp_rank_acc_3 += rank_acc_3
                disp_rank_acc_1 += rank_acc_1
                disp_rank_score += rank_score
                disp_pred_diff += pred_diff
                disp_edge_sign_acc += edge_sign_acc

                for i in range(len(goal_obj_names)):
                    if goal_idx[0] == i:
                        obj_results[goal_obj_names[i]]['count'] += 1
                        obj_results[goal_obj_names[i]]['disp_value_acc'] += value_acc
                        obj_results[goal_obj_names[i]]['disp_rank_acc_3'] += rank_acc_3
                        obj_results[goal_obj_names[i]]['disp_rank_acc_1'] += rank_acc_1
                        obj_results[goal_obj_names[i]]['disp_rank_score'] += rank_score
                        obj_results[goal_obj_names[i]]['disp_pred_diff'] += pred_diff
                        obj_results[goal_obj_names[i]]['disp_edge_sign_acc'] += edge_sign_acc


                if args.one_iter_test and not args.one_iter_test:
                    break


            end_time = time.time()

            val_metrics = {
                'epoch': epoch + 1,
                'val/loss': float(disp_loss / (cnt_in_val)),
                'val/value_loss': float(disp_value_loss / (cnt_in_val)),
                'val/adj_loss': float(disp_adj_loss / (cnt_in_val)),
                'val/adj_sim_loss': float(disp_adj_sim_loss / (cnt_in_val)),
                'val/sign_loss': float(disp_sign_loss / (cnt_in_val)),
                'val/value_acc': float(disp_value_acc / (cnt_in_val)),
                'val/rank_acc_3': float(disp_rank_acc_3 / (cnt_in_val)),
                'val/rank_acc_1': float(disp_rank_acc_1 / (cnt_in_val)),
                'val/rank_score': float(disp_rank_score / (cnt_in_val)),
                'val/pred_diff': float(disp_pred_diff / (cnt_in_val)),
                'val/edge_sign_acc': float(disp_edge_sign_acc / (cnt_in_val)),
            }

            print(
                f'Val [Epoch: {epoch + 1}/{args.max_epoch}],loss: {disp_loss / cnt_in_val:.5f}, '
                f'value_loss: {disp_value_loss / cnt_in_val:.5f}, adj_loss: {disp_adj_loss / cnt_in_val:.5f}, '
                f'adj_sim_loss: {disp_adj_sim_loss / cnt_in_val:.5f}, '
                f'sign_loss: {disp_sign_loss / cnt_in_val:.5f}, '
                f'value_acc: {disp_value_acc / (cnt_in_val):.5f}, '
                f'rank_acc_3: {disp_rank_acc_3 / (cnt_in_val):.5f}, '
                f'rank_acc_1: {disp_rank_acc_1 / (cnt_in_val):.5f}, '
                f'rank_score: {disp_rank_score / (cnt_in_val):.5f}, '
                f'pred_diff: {disp_pred_diff / (cnt_in_val):.5f}, '
                f'edge_sign_acc: {disp_edge_sign_acc / (cnt_in_val):.5f}, '
                f'lr: {lr_scheduler.get_lr()[0]:.5f}, time: {(end_time - start_time)//60:.0f}:{(end_time - start_time)%60:.0f}')

            for obj_name in goal_obj_names:
                cnt = obj_results[obj_name]['count']
                if cnt == 0: continue
                obj_disp_value_acc = obj_results[obj_name]['disp_value_acc']
                obj_disp_rank_acc_3 = obj_results[obj_name]['disp_rank_acc_3']
                obj_disp_rank_acc_1 = obj_results[obj_name]['disp_rank_acc_1']
                obj_disp_rank_score = obj_results[obj_name]['disp_rank_score']
                obj_disp_pred_diff = obj_results[obj_name]['disp_pred_diff']
                obj_disp_edge_sign_acc = obj_results[obj_name]['disp_edge_sign_acc']
                print(
                    f'     {obj_name} - value_acc: {obj_disp_value_acc / cnt:.5f}, '
                    f'rank_acc_3: {obj_disp_rank_acc_3 / cnt:.5f}, '
                    f'rank_acc_1: {obj_disp_rank_acc_1 / cnt:.5f}, '
                    f'rank_score: {obj_disp_rank_score / cnt:.5f}, '
                    f'pred_diff: {obj_disp_pred_diff / cnt:.5f}, '
                    f'edge_sign_acc: {obj_disp_edge_sign_acc / cnt:.5f}, '
                    f'cnt: {cnt}')


                val_metrics[f'val/obj_value_acc/{obj_name}'] = float(obj_disp_value_acc / cnt)
                val_metrics[f'val/obj_rank_acc_3/{obj_name}'] = float(obj_disp_rank_acc_3 / cnt)
                val_metrics[f'val/obj_rank_acc_1/{obj_name}'] = float(obj_disp_rank_acc_1 / cnt)
                val_metrics[f'val/obj_rank_score/{obj_name}'] = float(obj_disp_rank_score / cnt)
                val_metrics[f'val/obj_pred_diff/{obj_name}'] = float(obj_disp_pred_diff / cnt)
                val_metrics[f'val/obj_edge_sign_acc/{obj_name}'] = float(obj_disp_edge_sign_acc / cnt)



            if not 'test' in args.log_dir:
                wandb.log(val_metrics)


            torch.save(model.state_dict(), args.log_dir + '/model_{}.pth'.format(val_cnt))

            if float(disp_loss / (cnt_in_val)) < lowest_total_loss:
                lowest_total_loss = float(disp_loss / (cnt_in_val))
                torch.save(model.state_dict(), args.log_dir + f'/best_model_{val_cnt}.pth')

            # disp_loss = 0.0
            # disp_dist_acc = 0.0

            model.train()

    print('Finished Training')
    print(
        f'Val [loss: {disp_loss / cnt_in_val:.5f}, value_loss: {disp_value_loss / cnt_in_val:.5f}, '
        f'adj_loss: {disp_adj_loss / cnt_in_val:.5f}, '
        f'adj_sim_loss: {disp_adj_sim_loss / cnt_in_val:.5f}, '
        f'sign_loss: {disp_sign_loss / cnt_in_val:.5f}, '
        f'dist_acc: {disp_value_acc / (cnt_in_val):.5f}, '
        f'rank_acc_3: {disp_rank_acc_3 / (cnt_in_val):.5f}, '
        f'rank_acc_1: {disp_rank_acc_1 / (cnt_in_val):.5f}, '
        f'rank_score: {disp_rank_score / (cnt_in_val):.5f}, '
        f'pred_diff: {disp_pred_diff / (cnt_in_val):.5f}, '
        f'edge_sign_acc: {disp_edge_sign_acc / (cnt_in_val):.5f}, '
        f'lr: {lr_scheduler.get_lr()[0]:.5f}, time: {(time.time() - train_start_time) // 60:.0f}:{(time.time() - train_start_time) % 60:.0f}')

    for obj_name in goal_obj_names:
        cnt = obj_results[obj_name]['count']
        if cnt == 0:
            continue
        obj_disp_value_acc = obj_results[obj_name]['disp_value_acc']
        obj_disp_rank_acc_3 = obj_results[obj_name]['disp_rank_acc_3']
        obj_disp_rank_acc_1 = obj_results[obj_name]['disp_rank_acc_1']
        obj_disp_rank_score = obj_results[obj_name]['disp_rank_score']
        obj_disp_pred_diff = obj_results[obj_name]['disp_pred_diff']
        obj_disp_edge_sign_acc = obj_results[obj_name]['disp_edge_sign_acc']
        print(f'     {obj_name} - value_acc: {obj_disp_value_acc / cnt:.5f}, '
              f'rank_acc_3: {obj_disp_rank_acc_3 / cnt:.5f}, '
                f'rank_acc_1: {obj_disp_rank_acc_1 / cnt:.5f}, '
                f'rank_score: {obj_disp_rank_score / cnt:.5f}, '
                f'edge_sign_acc: {obj_disp_edge_sign_acc / cnt:.5f}, '
              f'pred_diff: {obj_disp_pred_diff / cnt:.5f}, cnt: {cnt}')


    print(args)

    with open(args.log_dir + '/result.txt', 'w') as f:
        f.write(str(args))
        f.write('\n')
        f.write(f'Val [loss: {disp_loss / cnt_in_val:.5f}, value_loss: {disp_value_loss / cnt_in_val:.5f}, '
                f'adj_loss: {disp_adj_loss / cnt_in_val:.5f}, '
                f'adj_sim_loss: {disp_adj_sim_loss / cnt_in_val:.5f}, '
                f'sign_loss: {disp_sign_loss / cnt_in_val:.5f}, '
                f'value_acc: {disp_value_acc / (cnt_in_val):.5f}, '
                f'rank_acc_3: {disp_rank_acc_3 / (cnt_in_val):.5f}, '
                f'rank_acc_1: {disp_rank_acc_1 / (cnt_in_val):.5f}, '
                f'rank_score: {disp_rank_score / (cnt_in_val):.5f}, '
                f'pred_diff: {disp_pred_diff / (cnt_in_val):.5f}, '
                f'edge_sign_acc: {disp_edge_sign_acc / (cnt_in_val):.5f}, '
                f'lr: {lr_scheduler.get_lr()[0]:.5f}, time: {(time.time() - train_start_time) // 60:.0f}:{(time.time() - train_start_time) % 60:.0f}\n')

        for obj_name in goal_obj_names:
            cnt = obj_results[obj_name]['count']
            disp_cnt = cnt
            if cnt == 0:
                cnt = 1

            obj_disp_value_acc = obj_results[obj_name]['disp_value_acc']
            obj_disp_rank_acc_3 = obj_results[obj_name]['disp_rank_acc_3']
            obj_disp_rank_acc_1 = obj_results[obj_name]['disp_rank_acc_1']
            obj_disp_rank_score = obj_results[obj_name]['disp_rank_score']
            obj_disp_pred_diff = obj_results[obj_name]['disp_pred_diff']
            obj_disp_edge_sign_acc = obj_results[obj_name]['disp_edge_sign_acc']
            f.write(f'     {obj_name} - value_acc: {obj_disp_value_acc / cnt:.5f}, '
                    f'rank_acc_3: {obj_disp_rank_acc_3 / cnt:.5f}, '
                    f'rank_acc_1: {obj_disp_rank_acc_1 / cnt:.5f}, '
                    f'rank_score: {obj_disp_rank_score / cnt:.5f}, '
                    f'pred_diff: {obj_disp_pred_diff / cnt:.5f}, '
                    f'edge_sign_acc: {obj_disp_edge_sign_acc / cnt:.5f}, '
                    f'cnt: {disp_cnt}\n')




def eval(checkpoint_path):
    use_gpu = torch.cuda.is_available()
    val_envs = [os.path.join(args.data_dir, 'val', name) for name in os.listdir(os.path.join(args.data_dir, 'val'))]
    val_envs.sort()

    if args.data_dir_aug is not None:
        for data_aug in args.data_dir_aug:
            val_envs_aug = [os.path.join(data_aug, 'val', name) for name in os.listdir(os.path.join(data_aug, 'val'))]
            val_envs_aug.sort()
            val_envs = val_envs + val_envs_aug


    val_list = []
    for env in val_envs:
        val_list = val_list + [os.path.join(env, x) for x in os.listdir(env)]
    val_list.sort()
    val_batch_num = int(len(val_list) / args.batch_size)
    val_num = val_batch_num * args.batch_size
    val_dataset = Batch_traj_DataLoader(args, val_list)

    def make_collate_batch(samples):
        node_features = torch.cat([sample['node_features'] for sample in samples], dim=0)
        node_info_features = torch.cat([sample['node_info_features'] for sample in samples], dim=0)
        node_goal_features = torch.cat([sample['node_goal_features'] for sample in samples], dim=0)
        node_goal_dists = torch.cat([sample['node_goal_dists'] for sample in samples], dim=0)
        node_pose = torch.cat([sample['node_pose'] for sample in samples], dim=0)
        goal_idx = torch.cat([sample['goal_idx'] for sample in samples], dim=0)

        batch_size = node_features.size()[0]
        adj_mtx = torch.zeros(batch_size, batch_size)
        min_max_mtx = torch.zeros(batch_size, batch_size)
        adj_starting_point = 0
        for i, sample in enumerate(samples):
            adj_size = sample['adj_mtx'].size()[0]
            adj_mtx[adj_starting_point:adj_starting_point + adj_size,
                    adj_starting_point:adj_starting_point + adj_size] = sample['adj_mtx']
            min_max_mtx[adj_starting_point:adj_starting_point + adj_size,
                        adj_starting_point:adj_starting_point + adj_size] = sample['min_max_mtx']
            
            adj_starting_point += adj_size
        return {
            'node_features': node_features,
            'node_info_features': node_info_features,
            'node_goal_features': node_goal_features,
            'adj_mtx': adj_mtx,
            'node_goal_dists': node_goal_dists,
            'node_pose': node_pose,
            'goal_idx': goal_idx,
            'min_max_mtx': min_max_mtx
        }

    valid_loader = DataLoader(dataset=val_dataset, batch_size=1, collate_fn=make_collate_batch, num_workers=4)

    if use_gpu:
        print("Currently using GPU {}".format(args.gpu))
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(args.seed)
        dev = "cuda:0"
    else:
        print("Currently using CPU")

    model = Model(args)
    model = nn.DataParallel(model).cuda()

    print("Model size: {:.5f}M".format(sum(p.numel() for p in model.parameters()) / 1000000.0))
    print("Loading checkpoint from '{}'".format(checkpoint_path))
    # model = torch.load(args.resume)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint)

    if use_gpu:
        model = nn.DataParallel(model).cuda()

    print("==> Start evaluation")

    # mse = nn.MSELoss()
    criterion = nn.MSELoss()
    # criterion = nn.KLDivLoss(reduction='batchmean')

    lowest_total_loss = 10000

    iter = 0
    val_cnt = 0
    train_start_time = time.time()



    if args.goal_cat == 'mp3d':
        goal_obj_names = mp3d_goal_obj_names
    elif args.goal_cat == 'mp3d_21':
        goal_obj_names = obj_names
    elif args.goal_cat == 'gibson':
        goal_obj_names = gibson_goal_obj_names

    obj_results = {}
    for obj_name in goal_obj_names:
        obj_results[obj_name] = {'count': 0, 'disp_value_acc': 0 , 'disp_rank_acc_3': 0, 'disp_rank_acc_1': 0,
                                 'disp_rank_score': 0, 'disp_pred_diff': 0, 'disp_edge_sign_acc':0}

    model.eval()
    with torch.no_grad():
        disp_loss = 0.0
        disp_value_loss = 0.0
        disp_adj_loss = 0.0
        disp_adj_sim_loss = 0.0
        disp_sign_loss = 0.0
        disp_value_acc = 0.0
        disp_rank_acc_3 = 0.0
        disp_rank_acc_1 = 0.0
        disp_rank_score = 0.0
        disp_pred_diff = 0.0
        disp_edge_sign_acc = 0.0
        cnt_in_val = 0
        start_time = time.time()
        # for i, data in enumerate(valid_loader, 0):
        true_value_cnt = np.zeros([10])
        pred_value_cnt = np.zeros([10])

        eyes = torch.eye(100).cuda()

        for data in tqdm(valid_loader, total=len(val_list)):

            features = data['node_features'].cuda()
            info_features = data['node_info_features'].cuda()
            goal_features = data['node_goal_features'].cuda()
            adj_mtx = data['adj_mtx'].cuda()
            node_goal_dists = data['node_goal_dists'].cuda()
            goal_idx = data['goal_idx']
            min_max_mtx = data['min_max_mtx'].cuda()

            cand_nodes = 1 - info_features[:, 0]
            if torch.sum(cand_nodes) < 3:
                continue

            # if len(adj_mtx) < 50:
            #     continue

            cnt_in_val += 1

            pred_dist = model(features, goal_features, info_features, adj_mtx)
            value_loss = criterion(pred_dist, node_goal_dists)

            ### for adj mtx
            if len(eyes) < len(adj_mtx):
                eyes = torch.eye(len(adj_mtx)).cuda()
            adj_mtx_dig0 = adj_mtx * (1 - eyes[:len(adj_mtx), :len(adj_mtx)])
            indices = torch.nonzero(adj_mtx_dig0, as_tuple=True)
            min_max_mtx0 = min_max_mtx * (1 - eyes[:len(adj_mtx), :len(adj_mtx)])
            min_max_indices = torch.nonzero(min_max_mtx0, as_tuple=True)
            adj_loss = criterion(pred_dist[indices[0]] - pred_dist[indices[1]],
                                 node_goal_dists[indices[0]] - node_goal_dists[indices[1]])
            if torch.isnan(adj_loss):
                adj_loss = torch.tensor(0.0).cuda()
            adj_sim_loss = criterion(pred_dist[min_max_indices[0]] - pred_dist[min_max_indices[1]], node_goal_dists[min_max_indices[0]] - node_goal_dists[min_max_indices[1]])
            if torch.isnan(adj_sim_loss):
                adj_sim_loss = torch.tensor(0.0).cuda()

            sign_loss = sign_penalty_loss(pred_dist[min_max_indices[0]] - pred_dist[min_max_indices[1]],
                                          node_goal_dists[min_max_indices[0]] - node_goal_dists[min_max_indices[1]])

            loss = args.value_loss_cf * value_loss + args.adj_loss_cf * adj_loss + args.sign_loss_cf * sign_loss + args.adj_sim_loss_cf * adj_sim_loss

            disp_loss += loss.item()
            disp_value_loss += args.value_loss_cf * value_loss.item()
            disp_adj_loss += args.adj_loss_cf * adj_loss.item()
            disp_adj_sim_loss += args.adj_sim_loss_cf * adj_sim_loss.item()
            disp_sign_loss += args.sign_loss_cf * sign_loss.item()

            value_acc = torch.mean(torch.where(abs(pred_dist - node_goal_dists) <= 0.1, 1, 0).float())
            cand_nodes = 1 - info_features[:, 0]
            if torch.sum(cand_nodes) == 0:
                topk_list = None
            elif torch.sum(cand_nodes) >= 3:
                topk_list = torch.topk(node_goal_dists[cand_nodes > 0], 3, dim=0).indices
            else:
                topk_list = torch.topk(node_goal_dists[cand_nodes > 0], int(torch.sum(1 - info_features[:, 0])),
                                       dim=0).indices

            if topk_list is None:
                rank_acc_3 = float(torch.Tensor([1]))
                rank_acc_1 = float(torch.Tensor([1]))
            else:
                rank_acc_3 = float(torch.argmax(pred_dist[cand_nodes > 0], dim=0) in topk_list)
                rank_acc_1 = float(torch.argmax(pred_dist[cand_nodes > 0], dim=0) in topk_list[0])

            oracle_rank = torch.topk(node_goal_dists[cand_nodes > 0], int(torch.sum(cand_nodes)), dim=0).indices[:, 0]
            rank_score = 1 - float((torch.where(oracle_rank == torch.argmax(pred_dist[cand_nodes > 0], dim=0))[0]) / torch.sum(cand_nodes))


            pred_diff = np.linalg.norm(data['node_pose'][torch.argmax(pred_dist)] - data['node_pose'][torch.argmax(node_goal_dists)])

            edge_sign_acc = float(torch.mean(((pred_dist[indices[0]] - pred_dist[indices[1]]) *
                                              (node_goal_dists[indices[0]] - node_goal_dists[indices[1]]) > 0
                                              ).float()))

            disp_value_acc += value_acc
            disp_rank_acc_3 += rank_acc_3
            disp_rank_acc_1 += rank_acc_1
            disp_pred_diff += pred_diff
            disp_rank_score += rank_score
            disp_edge_sign_acc += edge_sign_acc

            for i in range(len(goal_obj_names)):
                if goal_idx[0] == i:
                    obj_results[goal_obj_names[i]]['count'] += 1
                    obj_results[goal_obj_names[i]]['disp_value_acc'] += value_acc
                    obj_results[goal_obj_names[i]]['disp_rank_acc_3'] += rank_acc_3
                    obj_results[goal_obj_names[i]]['disp_rank_acc_1'] += rank_acc_1
                    obj_results[goal_obj_names[i]]['disp_rank_score'] += rank_score
                    obj_results[goal_obj_names[i]]['disp_pred_diff'] += pred_diff
                    obj_results[goal_obj_names[i]]['disp_edge_sign_acc'] += edge_sign_acc

            node_goal_dists_cnt = np.squeeze(np.array(node_goal_dists.cpu()) * 10, axis=1).astype(int)
            pred_dist_cnt = np.squeeze(np.array(pred_dist.cpu()) * 10, axis=1).astype(int)
            for i in range(len(node_goal_dists_cnt)):
                if node_goal_dists_cnt[i] > 9:
                    node_goal_dists_cnt[i] = 9
                if pred_dist_cnt[i] > 9:
                    pred_dist_cnt[i] = 9
                true_value_cnt[node_goal_dists_cnt[i]] += 1
                pred_value_cnt[pred_dist_cnt[i]] += 1

            if args.one_iter_test and not args.one_iter_test:
                break

    print('Finished evaluation')
    print(
        f'Val loss: {disp_loss / cnt_in_val:.5f}, value_loss: {disp_value_loss / cnt_in_val:.5f}, '
        f'adj_loss: {disp_adj_loss / cnt_in_val:.5f}, '
        f'adj_sim_loss: {disp_adj_sim_loss / cnt_in_val:.5f}, '
        f'sign_loss: {disp_sign_loss / cnt_in_val:.5f}, '
        f'value_acc: {disp_value_acc / (cnt_in_val):.5f}, rank_acc_3: {disp_rank_acc_3 / (cnt_in_val):.5f}, rank_acc_1: {disp_rank_acc_1 / (cnt_in_val):.5f}, '
        f'rank_score: {disp_rank_score / cnt_in_val:.5f}, pred_diff: {disp_pred_diff / (cnt_in_val):.5f}, edge_sign_acc: {disp_edge_sign_acc / (cnt_in_val):.5f}')

    for obj_name in goal_obj_names:
        cnt = obj_results[obj_name]['count']
        if cnt == 0: continue
        obj_disp_value_acc = obj_results[obj_name]['disp_value_acc']
        obj_disp_rank_acc_3 = obj_results[obj_name]['disp_rank_acc_3']
        obj_disp_rank_acc_1 = obj_results[obj_name]['disp_rank_acc_1']
        obj_disp_rank_score = obj_results[obj_name]['disp_rank_score']
        obj_disp_pred_diff = obj_results[obj_name]['disp_pred_diff']
        obj_disp_edge_sign_acc = obj_results[obj_name]['disp_edge_sign_acc']
        print(f'     {obj_name} - value_acc: {obj_disp_value_acc / cnt:.5f}, '
              f'rank_acc_3: {obj_disp_rank_acc_3 / cnt:.5f}, '
              f'rank_acc_1: {obj_disp_rank_acc_1 / cnt:.5f}, '
              f'rank_score: {obj_disp_rank_score / cnt:.5f}, '
                f'edge_sign_acc: {obj_disp_edge_sign_acc / cnt:.5f}, '
              f'pred_diff: {obj_disp_pred_diff / cnt:.5f}, cnt: {cnt}')


    print('true_value_cnt: ', true_value_cnt.astype(int))
    print('pred_value_cnt: ', pred_value_cnt.astype(int))
    print(args)


if __name__ == '__main__':

    main()
    # eval('/data1/hwing/Projects/offline_objgoal/goal_dist_pred/logs/cm_0610/0610_v2_1_use_cm_maxdist30.0_lr0.001/model_25.pth')
    # eval('/data1/hwing/Projects/offline_objgoal/goal_dist_pred/logs/cm_0616/0616_combv2_modelv2_1_use_cm_maxdist30.0_lr0.0001/model_20.pth')
    # eval('/home/hwing/Projects/offline_objgoal/goal_dist_pred/logs/cm_0701/0701_relpose_stepbystep_pano_goalscore_use_cm_maxdist30.0_lr0.01/model_20.pth')
    # eval('/home/hwing/Projects/offline_objgoal/goal_dist_pred/logs/cm_0706/0706_mp3d21_pano_goalscore_no_cm_maxdist30.0_lr0.01/model_17.pth')
    # eval('/home/hwing/Projects/OVG-Nav/goal_dist_pred/logs/cm_0706/0706_mp3d21_pano_goalscore_adjloss0.5_use_cm_maxdist30.0_lr0.01/model_10.pth')
    # eval('/home/hwing/Projects/OVG-Nav/goal_dist_pred/logs/cm_0712/0712_mp3d21_front_goalscore_adjloss0.5_use_cm_maxdist30.0_lr0.01/model_10.pth')
    # eval('/home/hwing/Projects/OVG-Nav/goal_dist_pred/logs/cm_0716/0716_mp3d21_panov2_goalscore_wadjmtx_adjloss0.5_use_cm_maxdist30.0_lr0.01/model_10.pth')
    # eval('/home/hwing/Projects/OVG-Nav/goal_dist_pred/logs/cm_2023-07-19/15-28_mp3d21_panov2_goalscore_wadjmtx_adjloss0.0_use_cm_maxdist30.0_lr0.01/model_10.pth')
    # eval('/home/hwing/Projects/OVG-Nav/goal_dist_pred/logs/cm_2023-07-21/02-12_mp3d21_panov2_goalscore_wadjmtx_valueloss_1.0_adjlossv2_100.0_use_cm_maxdist30.0_lr0.0001/model_10.pth')
    # eval('/home/hwing/Projects/OVG-Nav/goal_dist_pred/logs/cm_2023-07-20/20-35_mp3d21_panov2_goalscore_wadjmtx_valueloss_1.0_adjlossv2_100.0_use_cm_maxdist30.0_lr0.01/model_10.pth')
    # eval('/home/hwing/Projects/OVG-Nav/goal_dist_pred/models/01-11_mp3d21_edge1v2_panov3_2_layer10_hidden512_goalscore_w_adjmtx_valueloss1.0_adjloss1.0_10hoploss1.0_10hopsignloss0.001_use_cm_maxdist30.0_lr0.001/model_9.pth')
