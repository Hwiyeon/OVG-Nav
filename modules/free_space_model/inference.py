import torch
torch.set_num_threads(1)
import numpy as np

from torchvision import transforms
from modules.free_space_model.model_pano_free import Model_pano_split_free
from collections import OrderedDict


class FreeSpaceModel():
    def __init__(self, args):
        self.args = args
        self.device = f"cuda:{args.model_gpu}"

        self.model = Model_pano_split_free(args)
        state_dict = torch.load(args.free_space_model)
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]
            new_state_dict[name] = v
        self.model.load_state_dict(new_state_dict)
        self.model = self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        self.pano_width = args.pano_width
        self.pano_height = args.pano_height

    def get_dirc_imgs_from_pano(self, pano_img, num_imgs=12):

        width_bias = int((360/num_imgs)/2/360*self.pano_width)
        width_half = int(self.pano_height/2)

        # split the panorama into 12 square images with even angles
        dirc_imgs = []
        for i in range(num_imgs):
            angle = i * 360 / num_imgs
            x = int(self.pano_width  * (angle / 360)) + width_bias
            start_w = x - width_half
            end_w = x + width_half

            if start_w < 0:
                dirc_img = np.concatenate((pano_img[:, start_w:], pano_img[:, :end_w]), axis=1)
            elif end_w > self.pano_width :
                dirc_img = np.concatenate((pano_img[:, start_w:], pano_img[:, :end_w - self.pano_width ]), axis=1)
            else:
                dirc_img = pano_img[:, start_w:end_w]

            dirc_imgs.append(dirc_img)
        #         print(np.shape(dirc_img))
        return np.array(dirc_imgs)


    def predict_free_space(self, pano_rgb):
        dirc_rgbs = self.get_dirc_imgs_from_pano(pano_rgb)
        dirc_rgbs = dirc_rgbs.astype(np.float32) / 255.0
        pano_rgb = torch.stack([self.transform(rgb) for rgb in dirc_rgbs])
        pano_rgb = pano_rgb.to(self.device)

        with torch.no_grad():
            pano_rgb = torch.reshape(pano_rgb, [-1, 3, self.pano_height, self.pano_height])
            free_space = self.model(pano_rgb)
            free_space = torch.argmax(free_space, dim=1)

        free_space = torch.reshape(free_space, [-1]).detach().cpu().numpy()
        return free_space


if __name__ == '__main__':
    import argparse
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument('--free_space_model', type=str, default='ckpts/split_lr0.001_0227_range_2.0/best_model_1.pth')
    parser.add_argument('--model_gpu', type=int, default=9)
    parser.add_argument('--pano_width', type=int, default=512)
    parser.add_argument('--pano_height', type=int, default=128)
    args = parser.parse_args()

    free_space_model = FreeSpaceModel(args)
    pano_rgb = cv2.imread('test_data/pano_rgb.png')
    pano_rgb = cv2.cvtColor(pano_rgb, cv2.COLOR_BGR2RGB)
    free_space = free_space_model.predict_free_space(pano_rgb)
    print(free_space)

