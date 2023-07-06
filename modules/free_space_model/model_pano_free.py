import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
import math
import random

class Model_pano_free(nn.Module):
    def __init__(self, args):
        super(Model_pano_free, self).__init__()

        self.args = args

        self.pano_width = args.pano_width
        self.pano_height = args.pano_height

        self.reduced_width = self.pano_width
        self.reduced_height = self.pano_height
        for i in range(5):
            self.reduced_width = math.ceil(self.reduced_width/2)
            self.reduced_height = math.ceil(self.reduced_height/2)


        resnet = models.resnet50(pretrained=True)
        pred_hid = 512
        out_dim = args.feat_dim
        self.pred_hid = pred_hid
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.backbone_in_channels = resnet.fc.in_features

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 12))

        self.conv_base = nn.Sequential(
            nn.Conv2d(self.backbone_in_channels, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 2, kernel_size=3, stride=1, padding=1),
        )

    def get_single_feature(self, rgb):
        rgb = self.backbone(rgb)
        rgb = self.conv_base(rgb)
        rgb = self.avg_pool(rgb)
        return rgb

    def forward(self, pano_rgb):
        pano_rgb = self.get_single_feature(pano_rgb)

        return pano_rgb


class Model_pano_split_free(nn.Module):
    def __init__(self, args):
        super(Model_pano_split_free, self).__init__()

        self.args = args

        self.pano_width = args.pano_width
        self.pano_height = args.pano_height

        self.reduced_width = self.pano_width
        self.reduced_height = self.pano_height
        for i in range(5):
            self.reduced_width = math.ceil(self.reduced_width/2)
            self.reduced_height = math.ceil(self.reduced_height/2)


        resnet = models.resnet18(pretrained=True)
        pred_hid = 512
        self.pred_hid = pred_hid
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.backbone_in_channels = resnet.fc.in_features

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc_output = nn.Sequential(
            nn.Linear(self.backbone_in_channels, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Linear(512, 2),
        )

    def get_single_feature(self, rgb):
        rgb = self.backbone(rgb)
        rgb = torch.flatten(rgb, 1)
        rgb = self.fc_output(rgb)
        return rgb

    def forward(self, rgb):
        rgb = self.get_single_feature(rgb)

        return rgb