import os
os.environ["OMP_NUM_THREADS"] = '1'

# from model_pano_dirc_sim import Model
# from model_pano_dirc_sim import Model_feat as Model
from model_pano_free import Model_pano_split_free as Model

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import lr_scheduler
from torch.distributions import Bernoulli
from torch.utils.data import DataLoader
import torch.nn.functional as F
# from torchvision import transforms
import shutil

import numpy as np
import random
import time
import argparse
import sys
import torch.optim as optim
import wandb
from tqdm import tqdm


from dataloader_pano_free_0226 import RGB_pano_split_dataLoader_free as RGB_dataLoader
import matplotlib.pyplot as plt





parser = argparse.ArgumentParser("Pytorch code for unsupervised video summarization with REINFORCE")


parser.add_argument('--img-width', default=320, type=int)
parser.add_argument('--img-height', default=240, type=int)
parser.add_argument('--pano_width', default=512, type=int)
parser.add_argument('--pano_height', default=128, type=int)
parser.add_argument('--split', default=True, type=bool)
parser.add_argument('--rot_invariant', default=True, type=bool)
parser.add_argument('--use_depth', default=False, type=bool)
parser.add_argument('--feat_dim', default=256, type=int)
parser.add_argument('--sim_margin', type=float, default=0.8, help="learning rate (default: 1e-05)")
parser.add_argument('--free_range', default=1.0, type=float)
parser.add_argument('--use_resize', default=False, type=bool)

# Optimization options
parser.add_argument('--batch-size', type=int, default=256, help="learning rate (default: 1e-05)")
parser.add_argument('--lr', type=float, default=0.001, help="learning rate (default: 1e-05)")
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight_decay', type=float, default=1e-5)
parser.add_argument('--temperature', type=float, default=0.2)
parser.add_argument('--max-epoch', type=int, default=10, help="maximum epoch for training (default: 60)")
parser.add_argument('--stepsize', type=int, default=30, help="how many steps to decay learning rate (default: 30)")
parser.add_argument('--gamma', type=float, default=0.001, help="le arning rate decay (default: 0.1)")
parser.add_argument('--beta', type=float, default=0.01, help="weight for summary length penalty term (default: 0.01)")

# Misc
parser.add_argument('--seed', type=int, default=1, help="random seed (default: 1)")
parser.add_argument('--gpu', type=str, default='7', help="which gpu devices to use")
parser.add_argument('--use-cpu', action='store_true', help="use cpu device")
parser.add_argument('--evaluate', action='store_true', help="whether to do evaluation only")
parser.add_argument('--save-dir', type=str, default='log', help="path to save output (default: 'log/')")
parser.add_argument('--resume', type=str, default='', help="path to resume file")
parser.add_argument('--save-results', action='store_true', help="whether to save  output results")
parser.add_argument('--data-dir', default='/home/hwing/Dataset/habitat_obj_relation/mp3d_panoonly_free_0224_random_sample', type=str)
parser.add_argument('--log_dir', default='logs/0227/split_lr{}_0227_range_{}_no_resize', type=str)
parser.add_argument('--proj_name', default='0227_sim_pano_rgb_free', type=str)
parser.add_argument('--pcl_model_path', default='/disk1/hwing/Projects/offline_objgoal/models/PCL_rgbd_pano_256/PCL_encoder.pth', type=str)
parser.add_argument('--disp_iter', type=int, default=10, help="random seed (default: 1)")
parser.add_argument('--save_iter', type=int, default=3, help="random seed (default: 1)")
parser.add_argument('--checkpoints', type=str, default=None)

# --- code test ---
parser.add_argument('--one_iter_test', default=False, type=bool)


args = parser.parse_args()
''
print(args)
torch.manual_seed(args.seed)

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

# --- define logdir ---
if args.use_depth:
    args.log_dir = args.log_dir.format(args.lr, args.free_range)
else:
    args.log_dir = args.log_dir.format(args.lr, args.free_range)
# args.log_dir = args.log_dir.format(args.feat_dim, args.lr)
print(args.log_dir)

use_gpu = torch.cuda.is_available()
if args.use_cpu: use_gpu = False


train_data_dir = os.path.join(args.data_dir, 'train')
train_data_env_list = [os.path.join(train_data_dir, name, 'pano_rgb') for name in os.listdir(train_data_dir)]
train_data_list = []
for env in train_data_env_list:
    train_data_list += [os.path.join(env, name) for name in os.listdir(env)]
train_data_list = np.sort(train_data_list)

train_batch_num = int(len(train_data_list) / args.batch_size)
train_num = train_batch_num * args.batch_size
train_dataset = RGB_dataLoader(args, args.data_dir, train_data_list[:train_num], istrain=True, free_range=args.free_range, use_resize=args.use_resize)
train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)

val_data_dir = os.path.join(args.data_dir, 'val')
val_data_env_list = [os.path.join(val_data_dir, name, 'pano_rgb') for name in os.listdir(val_data_dir)]
val_data_list = []
for env in val_data_env_list:
    val_data_list += [os.path.join(env, name) for name in os.listdir(env)]
val_data_list = np.sort(val_data_list)

val_batch_num = int(len(val_data_list) / args.batch_size)
val_num = val_batch_num * args.batch_size
val_dataset = RGB_dataLoader(args, args.data_dir, val_data_list[:val_num], istrain=False, free_range=args.free_range, use_resize=args.use_resize)
valid_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, num_workers=2)



if not os.path.exists(args.log_dir):
    os.makedirs(args.log_dir)


def main():

    if use_gpu:
        print("Currently using GPU {}".format(args.gpu))
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(args.seed)
        dev = "cuda:0"
    else:
        print("Currently using CPU")

    model = Model(args)

    print("Model size: {:.5f}M".format(sum(p.numel() for p in model.parameters())/1000000.0))


    if args.resume:
        print("Loading checkpoint from '{}'".format(args.resume))
        model = torch.load(args.resume)
        # model.load_state_dict(checkpoint)
    else:
        start_epoch = 0

    if use_gpu:
        model = nn.DataParallel(model).cuda()


    if not 'test' in args.log_dir and args.one_iter_test == False:
        wandb.login(key='3e0c4716deb217aacb0f5f3991ac5c30637c8a7a')
        wandb.init(
            project=args.proj_name,
            name=args.log_dir.split('/')[-1],
        )
        wandb.config.update(args)




    print("==> Start training")
    epoch_start_time = time.time()
    start_time = time.time()


    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = lambda epoch: 0.999 ** epoch)
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(train_batch_num*args.max_epoch/3))
    cross_entropy = nn.CrossEntropyLoss()
    # triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)




    lowest_total_loss = 10000

    iter = 0
    val_cnt = 0
    for epoch in range(start_epoch, args.max_epoch):
        model.train()
        epoch_start_time = time.time()

        cnt = 0
        disp_loss = 0.0
        disp_acc = 0.0
        disp_acc_block = 0.0
        disp_acc_free = 0.0


        disp_iter = 0

        for i, data in enumerate(train_loader):
            disp_iter += 1
            pano_rgb1, label = data
            pano_rgb1, label = pano_rgb1.cuda(), label.cuda()

            pano_rgb1 = torch.reshape(pano_rgb1, [-1, 3, args.pano_height, args.pano_height])
            label = torch.reshape(label, [-1])
            feat_rgb1 = model(pano_rgb1)


            optimizer.zero_grad()
            # loss = constrative_loss(feat_rgb1, feat_rgb2)
            loss = cross_entropy(feat_rgb1, label)

            loss.backward()
            optimizer.step()

            disp_loss += loss.item()

            disp_acc += (torch.argmax(feat_rgb1, dim=1) == label).float().mean().item()
            disp_acc_block += torch.logical_and((torch.argmax(feat_rgb1, dim=1) == label), (label==0)).float().sum().item() / (label==0).sum().item()
            disp_acc_free += torch.logical_and((torch.argmax(feat_rgb1, dim=1) == label), (label==1)).float().sum().item() / (label==1).sum().item()



            cnt += 1
            iter += 1


            lr_scheduler.step()

            if cnt % args.disp_iter == 0:
                end_time = time.time()

                print(f'[Epoch: {epoch+1}/{args.max_epoch}] {cnt}/{train_batch_num} iter, '
                      f'loss: {disp_loss / disp_iter:.5f}, disp_acc: {disp_acc / disp_iter:.5f}, '
                      f'block_acc: {disp_acc_block / disp_iter:.5f}, free_acc: {disp_acc_free / disp_iter:.5f}, '
                      f'lr: {lr_scheduler.get_lr()[0]:.5f}, {args.disp_iter}iter time: {end_time - start_time:.3f}s, '
                      f'total time: {(end_time - epoch_start_time)//60:.0f}:{(end_time - epoch_start_time)%60:.0f}')


                metrics = {
                    'train/loss': float(disp_loss / (disp_iter)),
                    'train/acc': float(disp_acc / (disp_iter)),
                    'train/block_acc': float(disp_acc_block / (disp_iter)),
                    'train/free_acc': float(disp_acc_free / (disp_iter)),
                    'train/lr': lr_scheduler.get_lr()[0],

                    # 'train/loss_rot_aux': float(disp_loss_rot_aux / (disp_iter)),
                }
                if not 'test' in args.log_dir and args.one_iter_test == False:
                    wandb.log(metrics)


                disp_loss = 0.0
                disp_acc = 0.0
                disp_acc_block = 0.0
                disp_acc_free = 0.0
                disp_iter = 0

                start_time = time.time()

                if args.one_iter_test:
                    break



        val_cnt+=1
        model.eval()
        with torch.no_grad():
            disp_loss = 0.0
            disp_acc = 0.0
            disp_acc_block = 0.0
            disp_acc_free = 0.0


            cnt_in_val = 0
            epoch_start_time = time.time()
            start_time = time.time()
            for i, data in tqdm(enumerate(valid_loader), total=val_batch_num):
                disp_iter += 1
                pano_rgb1, label = data
                pano_rgb1, label = pano_rgb1.cuda(), label.cuda()

                pano_rgb1 = torch.reshape(pano_rgb1, [-1, 3, args.pano_height, args.pano_height])
                label = torch.reshape(label, [-1])

                feat_rgb1 = model(pano_rgb1)

                optimizer.zero_grad()
                loss = cross_entropy(feat_rgb1, label)

                disp_loss += loss.item()

                disp_acc += (torch.argmax(feat_rgb1, dim=1) == label).float().mean().item()
                disp_acc_block += torch.logical_and((torch.argmax(feat_rgb1, dim=1) == label), (label == 0)).float().sum().item() / (label == 0).sum().item()
                disp_acc_free += torch.logical_and((torch.argmax(feat_rgb1, dim=1) == label), (label == 1)).float().sum().item() / (label == 1).sum().item()

                cnt_in_val += 1

                if args.one_iter_test:
                    break


            end_time = time.time()
            print(
                f'Val [Epoch: {epoch + 1}/{args.max_epoch}], loss: {disp_loss / cnt_in_val:.5f}\n'
                f'acc: {disp_acc / cnt_in_val:.5f}, block_acc: {disp_acc_block / cnt_in_val:.5f}, free_acc: {disp_acc_free / cnt_in_val:.5f}\n'
                f'lr: {lr_scheduler.get_lr()[0]:.5f}, '
                f'total time: {(end_time - epoch_start_time)//60:.0f}:{(end_time - epoch_start_time)%60:.0f}')



            val_metrics = {
                'val/loss': float(disp_loss / (cnt_in_val)),
                'val/acc': float(disp_acc / (cnt_in_val)),
                'val/block_acc': float(disp_acc_block / (cnt_in_val)),
                'val/free_acc': float(disp_acc_free / (cnt_in_val)),
            }
            if not 'test' in args.log_dir and args.one_iter_test == False:
                wandb.log({**metrics, **val_metrics})


            torch.save(model.state_dict(), args.log_dir + '/model_{}.pth'.format(val_cnt))

            if float(disp_loss / (cnt_in_val)) < lowest_total_loss:
                lowest_total_loss = float(disp_loss / (cnt_in_val))
                torch.save(model.state_dict(), args.log_dir + f'/best_model_{val_cnt}.pth')

            model.train()
            start_time = time.time()



def vis_dataset():
    valid_loader = DataLoader(dataset=val_dataset, batch_size=1, num_workers=0)
    save_dir = 'test_data/1101'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    def load_data_to_image(data):
        data = data.cpu().squeeze().numpy()
        data = data.transpose(1, 2, 0)
        data = data[:,:,:3]
        data = (data * 255).astype(np.uint8)
        return data
    for i, data in tqdm(enumerate(valid_loader), total=val_batch_num):
        pano_rgbd1, pano_rgbd2, base_rgbd1, base_rgbd2, dirc_rgbd = data
        pano_rgbd1 = load_data_to_image(pano_rgbd1)
        pano_rgbd2 = load_data_to_image(pano_rgbd2)
        base_rgbd1 = load_data_to_image(base_rgbd1)
        base_rgbd2 = load_data_to_image(base_rgbd2)
        dirc_rgbd = load_data_to_image(dirc_rgbd)

        plt.subplot(3,3,1)
        plt.imshow(dirc_rgbd)
        plt.subplot(3,3,2)
        plt.imshow(base_rgbd1)
        plt.subplot(3,3,3)
        plt.imshow(base_rgbd2)
        plt.subplot(3,1,2)
        plt.imshow(pano_rgbd1)
        plt.subplot(3,1,3)
        plt.imshow(pano_rgbd2)
        plt.savefig(f'{save_dir}/sample_{i}.png')




if __name__ == '__main__':
    main()
    # vis_dataset()