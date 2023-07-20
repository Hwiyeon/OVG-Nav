#! /usr/bin/env python3
#
# %BANNER_BEGIN%
# ---------------------------------------------------------------------
# %COPYRIGHT_BEGIN%
#
#  Magic Leap, Inc. ("COMPANY") CONFIDENTIAL
#
#  Unpublished Copyright (c) 2020
#  Magic Leap, Inc., All Rights Reserved.
#
# NOTICE:  All information contained herein is, and remains the property
# of COMPANY. The intellectual and technical concepts contained herein
# are proprietary to COMPANY and may be covered by U.S. and Foreign
# Patents, patents in process, and are protected by trade secret or
# copyright law.  Dissemination of this information or reproduction of
# this material is strictly forbidden unless prior written permission is
# obtained from COMPANY.  Access to the source code contained herein is
# hereby forbidden to anyone except current COMPANY employees, managers
# or contractors who have executed Confidentiality and Non-disclosure
# agreements explicitly covering such access.
#
# The copyright notice above does not evidence any actual or intended
# publication or disclosure  of  this source code, which includes
# information that is confidential and/or proprietary, and is a trade
# secret, of  COMPANY.   ANY REPRODUCTION, MODIFICATION, DISTRIBUTION,
# PUBLIC  PERFORMANCE, OR PUBLIC DISPLAY OF OR THROUGH USE  OF THIS
# SOURCE CODE  WITHOUT THE EXPRESS WRITTEN CONSENT OF COMPANY IS
# STRICTLY PROHIBITED, AND IN VIOLATION OF APPLICABLE LAWS AND
# INTERNATIONAL TREATIES.  THE RECEIPT OR POSSESSION OF  THIS SOURCE
# CODE AND/OR RELATED INFORMATION DOES NOT CONVEY OR IMPLY ANY RIGHTS
# TO REPRODUCE, DISCLOSE OR DISTRIBUTE ITS CONTENTS, OR TO MANUFACTURE,
# USE, OR SELL ANYTHING THAT IT  MAY DESCRIBE, IN WHOLE OR IN PART.
#
# %COPYRIGHT_END%
# ----------------------------------------------------------------------
# %AUTHORS_BEGIN%
#
#  Originating Authors: Paul-Edouard Sarlin
#                       Daniel DeTone
#                       Tomasz Malisiewicz
#
# %AUTHORS_END%
# --------------------------------------------------------------------*/
# %BANNER_END%
import os
os.environ["OMP_NUM_THREADS"] = "1"

import argparse
import numpy as np
import torch
torch.set_num_threads(1)
import cv2
import quaternion

from modules.visual_odometry.models.matching import Matching
from modules.visual_odometry.models.utils import make_matching_plot_simple

torch.set_grad_enabled(False)


class KeypointMatching:
    def __init__(self, args):
        self.args = args
        self.width = args.KM_resize[0]
        self.height = args.KM_resize[1]
        self.hfov = args.front_hfov
        self.max_depth = args.max_depth
        self.min_depth = args.min_depth

        self.focal_length = (self.width / 2) / np.tan((self.hfov / 2) * np.pi / 180)
        self.camera_matrix = np.array([[self.focal_length, 0, self.width / 2],
                                       [0, self.focal_length, self.height / 2],
                                       [0, 0, 1]])

        self.step_size = args.move_forward
        self.act_rot = args.act_rot

        self.tgt_size = (args.KM_resize[0], args.KM_resize[1])
        self.tgt_hfov = args.front_hfov

        # --- keypoint matching model config ---
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = f'cuda:{args.model_gpu}'
        self.config = {
            'superpoint': {
                'nms_radius': args.KM_nms_radius,
                'keypoint_threshold': args.KM_keypoint_threshold,
                'max_keypoints': args.KM_max_keypoints
            },
            'superglue': {
                'weights': args.KM_superglue,
                'sinkhorn_iterations': args.KM_sinkhorn_iterations,
                'match_threshold': args.KM_match_threshold,
            }
        }

        self.kpts0 = None
        self.kpts1 = None

        self.model = Matching(self.config).to(self.device)
        self.model.eval()

    def resize_img(self, img, size=None):
        if size is None:
            size = (self.width, self.height)
        return cv2.resize(img, size)

    def rgb2gray(self, rgb):
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    def normalize_degree(self, degree):
        degree = int(np.round(degree)) % 360
        while True:
            if degree > 180:
                degree -= 360
            elif degree < -180:
                degree += 360
            else:
                break
        return degree

    def transform_input(self, rgb, depth=None):
        rgb = self.resize_img(rgb.astype('float32'), (self.width, self.height))
        rgb = self.rgb2gray(rgb)
        if depth is None:
            return rgb
        else:
            depth = self.resize_img(depth, (self.width, self.height))
            return rgb, depth

    def keypoint_matching(self, src_rgb, tgt_rgb, do_transform=False):
        src_rgb = torch.from_numpy(src_rgb / 255.).float()[None, None].to(self.device)
        tgt_rgb = torch.from_numpy(tgt_rgb / 255.).float()[None, None].to(self.device)

        pred = self.model({'image0': src_rgb, 'image1': tgt_rgb})
        pred = {k: v[0].cpu().numpy() for k, v in pred.items()}

        kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
        matches, conf = pred['matches0'], pred['matching_scores0']

        matching_kpts0 = kpts0[matches > -1]
        matching_kpts1 = kpts1[matches[matches > -1]]

        return matching_kpts0, matching_kpts1

    def get_ego_pc_from_pixel(self, keypoints, depth):
        """
        keypoints: [N, 2] array of (x, y) pixel (image) coordinates
        depth: [H, W] array of depth values

        returns: [N, 3] array of (x, y, z) coordinates in the camera coord
        """
        pc = []
        for i, point in enumerate(keypoints):
            ix = point[0]
            iy = point[1]
            z = depth[int(iy)][int(ix)]

            cx = (ix - int(self.width / 2)) / self.focal_length * z
            cy = (iy - int(self.height / 2)) / self.focal_length * z
            cz = z

            pc.append([cx, cy, cz])

        return np.array(pc).astype(np.float32)

    def get_extrinsic_guess_from_action(self, action):
        if type(action) == str:
            if action == 'move_forward':
                extrinsic_guess = np.array([[0, 0, -0.25], [0, 0, 0]]).astype(np.float32)
            elif action == 'turn_left':
                extrinsic_guess = np.array([[0, 0, 0], [0, self.act_rot * np.pi / 180., 0]]).astype(np.float32)
            elif action == 'turn_right':
                extrinsic_guess = np.array([[0, 0, 0], [0, -self.act_rot * np.pi / 180., 0]]).astype(np.float32)
        elif type(action) == int:
            if action == 1:
                extrinsic_guess = np.array([[0, 0, -0.25], [0, 0, 0]]).astype(np.float32)
            elif action == 2:
                extrinsic_guess = np.array([[0, 0, 0], [0, self.act_rot * np.pi / 180., 0]]).astype(np.float32)
            elif action == 3:
                extrinsic_guess = np.array([[0, 0, 0], [0, -self.act_rot * np.pi / 180., 0]]).astype(np.float32)

        guess = {
            'rvec': extrinsic_guess[1],
            'tvec': extrinsic_guess[0]
        }
        return guess

    def get_camera_transform(self, pc0, keypoints1, extrinsic_guess=None, rot_vec=False):
        """
        pc0: (N, 3) depth expansions of keypoints on the src camera (src camera coord)
        keypoints1: (N, 2) keypoints on the tgt camera (tgt image coord)

        return : transformations from src camera to tgt camera (habtiata coord)
        camera coord -> habitat coord : x -> x, y -> -y, z -> -z
        """
        keypoint_est_success = True
        np.random.seed()
        try:
            if not extrinsic_guess is None:
                # _, rvec, tvec, _ = cv2.solvePnPRansac(pc0, keypoints1, self.camera_matrix, distCoeffs=None,
                #                                       reprojectionError=1.0, iterationsCount=1000,
                #                                       useExtrinsicGuess=True,
                #                                       flags=cv2.SOLVEPNP_ITERATIVE)
                _, rvec, tvec, _ = cv2.solvePnPRansac(pc0, keypoints1, self.camera_matrix, distCoeffs=None,
                                                      reprojectionError=1.0, iterationsCount=1000,
                                                      rvec=extrinsic_guess['rvec'], tvec=extrinsic_guess['tvec'],
                                                      useExtrinsicGuess=True,
                                                      flags=cv2.SOLVEPNP_ITERATIVE)

                # rvec, tvec = cv2.solvePnPRefineLM(pc0, keypoints1, self.camera_matrix, distCoeffs=None,
                #                                       rvec=extrinsic_guess['rvec'], tvec=extrinsic_guess['tvec'])
            else:
                _, rvec, tvec, _ = cv2.solvePnPRansac(pc0, keypoints1, self.camera_matrix, distCoeffs=None,
                                                      reprojectionError=1.0, iterationsCount=1000,
                                                      useExtrinsicGuess=True,
                                                      flags=cv2.SOLVEPNP_ITERATIVE)
        except:
            # print('what happend?')
            keypoint_est_success = False
            rvec = extrinsic_guess['rvec']
            tvec = extrinsic_guess['tvec']
        ## rvec, tvec : object transformation from src camera coord to tgt camera coord --> reverse of the camera transformation

        est_rot_diff = np.squeeze(-rvec[1]) * 180 / np.pi  # - : for reverse transformation,
        est_rot_diff = self.normalize_degree(est_rot_diff)

        # if (np.linalg.norm(tvec) > self.step_size * 2 or est_rot_diff > self.act_rot * 2):
        #     print(rvec, tvec)
        #     print(extrinsic_guess['rvec'], extrinsic_guess['tvec'], '\n')
        extrinsic_rot_diff = self.normalize_degree(np.squeeze(-extrinsic_guess['rvec'][1]) * 180 / np.pi)

        reprojectionError = 1.0
        # while (np.linalg.norm(tvec) > self.step_size * 2 or abs(est_rot_diff) > self.act_rot * 2):
        while True:
            if np.linalg.norm(tvec) < self.step_size and abs(est_rot_diff - extrinsic_rot_diff) < self.act_rot / 2:
                break
            # elif np.linalg.norm(tvec-extrinsic_guess['tvec']) < self.step_size / 2 and abs(est_rot_diff) < self.act_rot:
            #     break
            elif np.linalg.norm(tvec-extrinsic_guess['tvec']) < self.step_size /2 and abs(est_rot_diff - extrinsic_rot_diff) < self.act_rot / 2:
                break

            if reprojectionError > 8.0:
                keypoint_est_success = False
                rvec = extrinsic_guess['rvec']
                tvec = extrinsic_guess['tvec']

                est_rot_diff = np.squeeze(-rvec[1]) * 180 / np.pi  # - : for reverse transformation,
                est_rot_diff = self.normalize_degree(est_rot_diff)
                break

            _, rvec, tvec, _ = cv2.solvePnPRansac(pc0, keypoints1, self.camera_matrix, distCoeffs=None,
                                                  reprojectionError=reprojectionError, iterationsCount=10000,
                                                  useExtrinsicGuess=True,
                                                  rvec=extrinsic_guess['rvec'], tvec=extrinsic_guess['tvec'],
                                                  flags=cv2.SOLVEPNP_ITERATIVE)

            est_rot_diff = np.squeeze(-rvec[1]) * 180 / np.pi  # - : for reverse transformation,
            est_rot_diff = self.normalize_degree(est_rot_diff)
            reprojectionError *= 2

        est_pos_diff = -tvec.reshape(-1)  # - : for reverse transformation

        #  for habitat coord
        # est_rot_diff = -est_rot_diff
        est_pos_diff = np.array([est_pos_diff[0], -est_pos_diff[1], -est_pos_diff[2]])  ## camera coord -> habitat coord
        if rot_vec:
            est_rot_diff = -rvec.reshape(-1)
            est_rot_diff = np.array([est_rot_diff[0], -est_rot_diff[1], -est_rot_diff[2]])
            return est_pos_diff, est_rot_diff, keypoint_est_success
        else:
            est_rot_diff = -est_rot_diff
            return est_pos_diff, est_rot_diff, keypoint_est_success

    def get_relative_camera_pose(self, src_rgb, src_depth, tgt_rgb, extrinsic_guess=None, rot_vec=False):
        """
        src_rgb: [H, W, 3] array of RGB values
        src_depth: [H, W] array of depth values
        tgt_rgb: [H, W, 3] array of RGB values

        returns: (position, rotation) tuple of camera transformation from src to tgt
        """
        src_rgb, src_depth = self.transform_input(src_rgb, src_depth)
        tgt_rgb = self.transform_input(tgt_rgb)

        kpts0, kpts1 = self.keypoint_matching(src_rgb, tgt_rgb)

        # --- get depth expansion of keypoints on the src camera ---
        pc0 = self.get_ego_pc_from_pixel(kpts0, src_depth)

        # --- get camera transformation from src camera to tgt camera ---
        est_pos_diff, est_rot_diff, keypoint_est_success = self.get_camera_transform(pc0, kpts1,
                                                                                     extrinsic_guess=extrinsic_guess,
                                                                                     rot_vec=rot_vec)

        return est_pos_diff, est_rot_diff, keypoint_est_success

    def get_matched_keypoints(self, src_rgb, src_depth, tgt_rgb):
        """
        src_rgb: [H, W, 3] array of RGB values
        src_depth: [H, W] array of depth values
        tgt_rgb: [H, W, 3] array of RGB values

        returns: (position, rotation) tuple of camera transformation from src to tgt
        """
        src_rgb, src_depth = self.transform_input(src_rgb, src_depth)
        tgt_rgb = self.transform_input(tgt_rgb)

        kpts0, kpts1 = self.keypoint_matching(src_rgb, tgt_rgb)

        return kpts0, kpts1

    def viz_matching(self, image0, image1, mkpts0, mkpts1, path=None):
        text = [
            'SuperGlue',
            'Matches: {}'.format(len(mkpts0)),
        ]

        image0 = self.resize_img(image0)
        image1 = self.resize_img(image1)

        make_matching_plot_simple(
            image0, image1, mkpts0, mkpts1, text, path=path)



def main():
    import os
    from tqdm import tqdm
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(
        description='Image pair matching and pose evaluation with SuperGlue',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        '--hfov', type=int, default=70,
        help='camera horizontal field of view')
    parser.add_argument(
        '--max_depth', type=float, default=10.,
        help='maximum depth value')
    parser.add_argument(
        '--min_depth', type=float, default=0.1,
        help='minimum depth value')

    # VO setttings
    parser.add_argument("--vo_width", type=int, default=320)
    parser.add_argument("--vo_height", type=int, default=240)
    parser.add_argument("--vo_hfov", type=int, default=70)

    parser.add_argument(
        '--KM_resize', type=int, nargs='+', default=[640, 480],
        help='Resize the input image before running inference. If two numbers, '
             'resize to the exact dimensions, if one number, resize the max '
             'dimension, if -1, do not resize')
    parser.add_argument(
        '--KM_superglue', choices={'indoor', 'outdoor'}, default='indoor',
        help='SuperGlue weights')
    parser.add_argument(
        '--KM_max_keypoints', type=int, default=1024,
        help='Maximum number of keypoints detected by Superpoint'
             ' (\'-1\' keeps all keypoints)')
    parser.add_argument(
        '--KM_keypoint_threshold', type=float, default=0.005,
        help='SuperPoint keypoint detector confidence threshold')
    parser.add_argument(
        '--KM_nms_radius', type=int, default=4,
        help='SuperPoint Non Maximum Suppression (NMS) radius'
             ' (Must be positive)')
    parser.add_argument(
        '--KM_sinkhorn_iterations', type=int, default=20,
        help='Number of Sinkhorn iterations performed by SuperGlue')
    parser.add_argument(
        '--KM_match_threshold', type=float, default=0.2,
        help='SuperGlue match threshold')

    parser.add_argument(
        '--viz',
        default=True,
        # action='store_true',
        help='Visualize the matches and dump the plots')

    parser.add_argument(
        '--fast_viz', action='store_true',
        help='Use faster image visualization with OpenCV instead of Matplotlib')
    parser.add_argument(
        '--show_keypoints', action='store_true',
        help='Plot the keypoints in addition to the matches')

    parser.add_argument(
        '--opencv_display', action='store_true',
        help='Visualize via OpenCV before saving output images')

    parser.add_argument("--move_forward", type=float, default=0.25)
    parser.add_argument("--act_rot", type=int, default=10)

    opt = parser.parse_args()
    print(opt)

    keypoint_matching = KeypointMatching(opt)

    data_dir = '/home/hwing/Projects/offline_objgoal/models/visual_odometry/test_data/data'
    save_dir = '/home/hwing/Projects/offline_objgoal/models/visual_odometry/test_data/results'

    epi_name = '2azQ1b91cZZ_0000'
    traj_data_path = f'{data_dir}/{epi_name}/{epi_name}.npy'
    rgb_data_path = f'{data_dir}/{epi_name}/rgb.avi'
    depth_data_path = f'{data_dir}/{epi_name}/depth.avi'

    if opt.viz:
        save_dir_path = f'{save_dir}/{epi_name}'
        if not os.path.exists(save_dir_path):
            os.makedirs(save_dir_path)

    traj_data = np.load(traj_data_path, allow_pickle=True).item()

    rgb_list = []
    cap = cv2.VideoCapture(rgb_data_path)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        rgb_list.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    depth_list = []
    cap = cv2.VideoCapture(depth_data_path)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        depth_list.append(frame[:, :, 0] / 255. * 10.)

    err_pos_diff_list = []
    err_rot_diff_list = []

    def vis_imgs(save_dir_path, src_img, tgt_img, epi_name, frame_idx, gt_data=None, est=None):
        # src_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB)
        # tgt_img = cv2.cvtColor(tgt_img, cv2.COLOR_BGR2RGB)

        plt.subplots(1, 2, figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(src_img)
        plt.subplot(1, 2, 2)
        plt.imshow(tgt_img)

        if gt_data:
            action = gt_data['action'][frame_idx]
            gt_pos_diff = gt_data['position'][frame_idx + 1] - gt_data['position'][frame_idx]
            gt_pos_diff = quaternion.rotate_vectors(gt_data['rotation'][frame_idx].inverse(), gt_pos_diff)
            gt_rot_diff = gt_data['rotation'][frame_idx + 1] * gt_data['rotation'][frame_idx].inverse()
            gt_rot_diff = quaternion.as_rotation_vector(gt_rot_diff)[1] * 180 / np.pi

            est_pos_diff = est['position_diff']
            est_rot_diff = est['rotation_diff']

            err_pos_diff = sum(abs(gt_pos_diff - est_pos_diff))
            err_rot_diff = abs(gt_rot_diff - est_rot_diff)

            plt.suptitle(f'Action: {action} \n'
                         f'GT: {gt_pos_diff[0]:.2f}, {gt_pos_diff[1]:.2f}, {gt_pos_diff[2]:.2f}, {gt_rot_diff:.2f} \n '
                         f'EST: {est_pos_diff[0]:.2f}, {est_pos_diff[1]:.2f}, {est_pos_diff[2]:.2f}, {est_rot_diff:.2f} \n'
                         f'ERR: {err_pos_diff:.2f}, {err_rot_diff:.2f} ({err_rot_diff * np.pi / 180.:.2f} rad)')

        plt.savefig(f'{save_dir_path}/{epi_name}_{frame_idx:04d}_{epi_name}_{frame_idx + 1:04d}.png')
        plt.close()



    for frame_idx in tqdm(range(len(depth_list) - 1)):

        # if traj_data['action'][frame_idx] == 'move_forward':
        #     gt_diff_action = np.array([[0, 0, -0.25], [0, 0, 0]]).astype(np.float32)
        #
        # elif traj_data['action'][frame_idx] == 'turn_left':
        #     gt_diff_action = np.array([[0, 0, 0], [0, 10 * np.pi / 180., 0]]).astype(np.float32)
        #
        # elif traj_data['action'][frame_idx] == 'turn_right':
        #     gt_diff_action = np.array([[0, 0, 0], [0, -10 * np.pi / 180., 0]]).astype(np.float32)

        extrinsic_guess = keypoint_matching.get_extrinsic_guess_from_action(traj_data['action'][frame_idx])


        est_pos_diff, est_rot_diff, keypoint_est_success = keypoint_matching.get_relative_camera_pose(rgb_list[frame_idx],
                                                                                depth_list[frame_idx],
                                                                                rgb_list[frame_idx + 1],
                                                                                extrinsic_guess=extrinsic_guess,
                                                                                rot_vec=True,
                                                                                )
        est_diff = {
            'position_diff': est_pos_diff,
            'rotation_diff': est_rot_diff
        }

        ## gt transform
        gt_pos_diff = traj_data['position'][frame_idx + 1] - traj_data['position'][frame_idx]
        gt_pos_diff = quaternion.rotate_vectors(traj_data['rotation'][frame_idx].inverse(), gt_pos_diff)

        gt_rot_diff = traj_data['rotation'][frame_idx + 1] * traj_data['rotation'][frame_idx].inverse()
        gt_rot_diff = quaternion.as_rotation_vector(gt_rot_diff)[1] * 180 / np.pi

        # if opt.viz:
        #     vis_imgs(save_dir_path, rgb_list[frame_idx], rgb_list[frame_idx + 1], epi_name, frame_idx,
        #              gt_data=traj_data, est=est_diff)

        err_pos_diff_list.append(sum(abs(gt_pos_diff - est_pos_diff)))
        err_rot_diff_list.append(abs(gt_rot_diff - est_rot_diff))

    print(f'Position error: {sum(err_pos_diff_list) / len(err_pos_diff_list)}')
    print(f'Rotation error - Degrees {sum(err_rot_diff_list) / len(err_rot_diff_list)} \n'
          f'               - Radians {sum(err_rot_diff_list) / len(err_rot_diff_list) * np.pi / 180.}')



if __name__ == '__main__':
    main()

