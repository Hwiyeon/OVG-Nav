from torch.utils.data import Dataset
from torchvision import transforms
import torch
import cv2
from PIL import Image
import random
import numpy as np
from utils.obj_category_info import assign_room_category
import os
import matplotlib.pyplot as plt
import json
import os

from habitat_sim.utils.common import (
    d3_40_colors_rgb,
    download_and_unzip,
    quat_from_angle_axis,
)



import numbers
from collections.abc import Sequence
from torch import Tensor

# from detector.detector_mask import Detector


transform = transforms.Compose([  # [1]
        transforms.Normalize(  # [5]
            mean=[0.485, 0.456, 0.406],  # [6]
            std=[0.229, 0.224, 0.225]  # [7]
        )])


class RGB_pano_dataLoader_free(Dataset):
    def __init__(self, args, data_dir, data_list, istrain=False, free_range=1.0, use_resize=True):
        self.args = args

        self.data_list = data_list
        self.len_data = len(self.data_list)

        if istrain:
            self.data_type = 'train'
        else:
            self.data_type = 'val'

        self.data_dir = os.path.join(data_dir, self.data_type)
        self.use_depth = args.use_depth


        # self.pano_len = len(os.listdir(os.path.join(self.data_dir, 'pano_rgb')))
        # self.pano_names = os.listdir(os.path.join(self.data_dir, 'pano_rgb'))

        self.istrain = istrain
        self.free_range = free_range
        self.use_resize = use_resize

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        self.rgb_augmentation = transforms.Compose([
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            # transforms.ColorJitter(brightness=0., contrast=0., saturation=0., hue=0.),
            # transforms.GaussianBlur(3, sigma=(0.1, 2.0)),
        ])

        self.flip_augmentation = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
        ])


        self.pano_size = (args.pano_height, args.pano_width)
        self.dirc_size = (args.img_height, args.img_width)

        self.pano_resize = transforms.Resize((int(self.pano_size[0]/2), int(self.pano_size[1]/2)))

        self.depth_scale = np.iinfo(np.uint16).max


    def __getitem__(self, index):
        pano_rgb, free_angle = self.load_data_pano(self.data_list[index])
        return pano_rgb, free_angle


    def __len__(self):
        return len(self.data_list)


    def load_data_pano(self, data_path):
        """
        :param args:
            --> 1 = forward, 2 = rot_left, 3 = rot_right
        :return: [demon_rgb, demon_action, trial_rgb, trial_action, action_mask]
        """

        # pano_name = data_path['pano']

        # --- load images ---
        pano_rgb1 = cv2.imread(data_path)
        pano_rgb1 = cv2.cvtColor(pano_rgb1, cv2.COLOR_BGR2RGB)
        if self.use_depth:
            pano_depth1 = cv2.imread(data_path, cv2.IMREAD_ANYDEPTH) / self.depth_scale


        # # --- transformation ---
        # angle = random.randint(0, np.shape(pano_rgb1)[1] - 1)
        # pano_rgb2 = np.roll(pano_rgb1, angle, axis=1)
        # if self.use_depth:
        #     pano_depth2 = np.roll(pano_depth1, angle, axis=1)

        pano_rgb1 = self.transform(pano_rgb1)
        if self.use_resize:
            pano_rgb1 = self.pano_resize(pano_rgb1)

        free_angle_path = data_path.replace('pano_rgb', 'pano_free_angle').replace('.png', '.npy')
        free_angle = np.load(free_angle_path, allow_pickle=True).item()
        free_angle_range = free_angle[f'free_cand_node_{self.free_range}']
        # label = torch.zeros([2, 1, 12]).int()
        # for i, angle in enumerate(free_angle):
        #     label[int(angle), 0, i] = 1

        label = torch.Tensor(free_angle_range).long().unsqueeze(0)



        return pano_rgb1.float(), label



class RGB_pano_split_dataLoader_free(Dataset):
    def __init__(self, args, data_dir, data_list, istrain=False, free_range=1.0, use_resize=True):
        self.args = args

        self.data_list = data_list
        self.len_data = len(self.data_list)

        if istrain:
            self.data_type = 'train'
        else:
            self.data_type = 'val'

        self.data_dir = os.path.join(data_dir, self.data_type)
        self.use_depth = args.use_depth


        # self.pano_len = len(os.listdir(os.path.join(self.data_dir, 'pano_rgb')))
        # self.pano_names = os.listdir(os.path.join(self.data_dir, 'pano_rgb'))

        self.istrain = istrain
        self.free_range = free_range
        self.use_resize = use_resize

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        self.rgb_augmentation = transforms.Compose([
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            # transforms.ColorJitter(brightness=0., contrast=0., saturation=0., hue=0.),
            # transforms.GaussianBlur(3, sigma=(0.1, 2.0)),
        ])

        self.flip_augmentation = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
        ])


        self.pano_size = (args.pano_height, args.pano_width)
        self.dirc_size = (args.img_height, args.img_width)

        self.pano_resize = transforms.Resize((int(self.pano_size[0]/2), int(self.pano_size[1]/2)))

        self.depth_scale = np.iinfo(np.uint16).max


    def __getitem__(self, index):
        pano_rgb, free_angle = self.load_data_pano(self.data_list[index])
        return pano_rgb, free_angle


    def __len__(self):
        return len(self.data_list)

    def get_dirc_imgs_from_pano(self, pano_img, pw, ph, num_imgs=12):

        width_bias = int((360/num_imgs)/2)
        width_half = int(ph/2)

        # split the panorama into 12 square images with even angles
        dirc_imgs = []
        for i in range(num_imgs):
            angle = i * 360 / num_imgs
            x = int(pw * (angle / 360)) + width_bias
            start_w = x - width_half
            end_w = x + width_half

            if start_w < 0:
                dirc_img = np.concatenate((pano_img[:, start_w:], pano_img[:, :end_w]), axis=1)
            elif end_w > pw:
                dirc_img = np.concatenate((pano_img[:, start_w:], pano_img[:, :end_w - pw]), axis=1)
            else:
                dirc_img = pano_img[:, start_w:end_w]

            dirc_imgs.append(dirc_img)
        #         print(np.shape(dirc_img))
        return np.array(dirc_imgs)

    def load_data_pano(self, data_path):
        """
        :param args:
            --> 1 = forward, 2 = rot_left, 3 = rot_right
        :return: [demon_rgb, demon_action, trial_rgb, trial_action, action_mask]
        """

        # pano_name = data_path['pano']

        # --- load images ---
        pano_rgb1 = cv2.imread(data_path)
        pano_rgb1 = cv2.cvtColor(pano_rgb1, cv2.COLOR_BGR2RGB)
        if self.use_depth:
            pano_depth1 = cv2.imread(data_path, cv2.IMREAD_ANYDEPTH) / self.depth_scale


        # # --- transformation ---
        # angle = random.randint(0, np.shape(pano_rgb1)[1] - 1)
        # pano_rgb2 = np.roll(pano_rgb1, angle, axis=1)
        # if self.use_depth:
        #     pano_depth2 = np.roll(pano_depth1, angle, axis=1)

        dirc_rgbs = self.get_dirc_imgs_from_pano(pano_rgb1, np.shape(pano_rgb1)[1], np.shape(pano_rgb1)[0])

        pano_rgb1 = torch.stack([self.transform(rgb) for rgb in dirc_rgbs])
        # if self.use_resize:
        #     pano_rgb1 = self.pano_resize(pano_rgb1)

        free_angle_path = data_path.replace('pano_rgb', 'pano_free_angle').replace('.png', '.npy')
        free_angle = np.load(free_angle_path, allow_pickle=True).item()
        free_angle_range = free_angle[f'free_cand_node_{self.free_range}']
        # label = torch.zeros([2, 1, 12]).int()
        # for i, angle in enumerate(free_angle):
        #     label[int(angle), 0, i] = 1

        label = torch.Tensor(free_angle_range).long().unsqueeze(0)



        return pano_rgb1.float(), label
