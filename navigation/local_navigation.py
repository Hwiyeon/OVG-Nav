from validity_func.validity_utils import (
    get_relative_location,
    get_sim_location,
)
from validity_func.local_nav import LocalAgent

import os
import numpy as np
from scipy.spatial.transform import Rotation as R
import quaternion
import math
import cv2

import visualizations as vis

import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

class LocalNavigation(object):
    def __init__(self, args, vo_module=None):
        self.args = args
        # self.agent = agent
        self.vo_module = vo_module
        self.local_agent = LocalAgent(args)

        self.action_idx_map = ['stop', 'move_forward', 'turn_left', 'turn_right']

    def reset_with_curr_pose(self, curr_pos, curr_rot, sim=None, agent=None):
        ## curr_rot : rotation vector
        # curr_rot = -np.array([curr_rot[0], -curr_rot[1], -curr_rot[1]])

        self.local_agent.reset_with_curr_pose(curr_pos, curr_rot)
        if sim is not None:
            self.sim = sim
        if agent is not None:
            self.agent = agent
        self.agent_steps = 0
        self.agent_length_taken = 0

    def reset_sim(self, sim):
        self.sim = sim

    def reset_agent(self, agent):
        self.agent = agent

    def reset_sim_and_agent(self, sim, agent):
        self.sim = sim
        self.agent = agent


    def get_vo_relative_camera_pose(self, prev_rgb, prev_depth, curr_rgb, action, initial_guess=None):
        if initial_guess is None:
            initial_guess = self.vo_module.get_extrinsic_guess_from_action(action)
        est_pos_diff, est_rot_diff, keypoint_est_success = self.vo_module.get_relative_camera_pose(
                                                            prev_rgb, prev_depth, curr_rgb, initial_guess, rot_vec=True)

        return est_pos_diff, est_rot_diff, keypoint_est_success


    def navigate_to_goal_point(self, goal_point, obs, use_est_map=True, use_gt_map=False,
                               visualize=False, vis_save_dir=None, vis_png=False, vis_video=False):

        assert use_est_map or use_gt_map, "At least one of use_est_map or use_gt_map should be True"
        if visualize:
            use_est_map, use_gt_map = True, True

        delta_dist, delta_rot = get_relative_location(self.local_agent.curr_pos,
                                                      self.local_agent.curr_rot,
                                                      goal_point)
        self.local_agent.update_gt_local_map(obs['depth_sensor'])
        self.local_agent.update_est_local_map(obs['depth_sensor'])
        self.local_agent.set_goal(delta_dist, delta_rot)
        prev_rgb = obs['color_sensor']
        prev_depth = obs['depth_sensor']

        curr_position = self.local_agent.curr_pos
        curr_rotation = self.local_agent.curr_rot

        if vis_video:
            frames = []

        # try:
        for idx in range(50):
            action, terminate_local = self.local_agent.navigate_local()
            action = self.action_idx_map[action]
            obs = self.sim.step(action)
            curr_rgb = obs['color_sensor']
            curr_depth = obs["depth_sensor"]  # not normalized
            if obs['collided']:
                self.local_agent.collision = True

            depth_m = curr_depth

            if use_gt_map:
                # --- GT pose
                gt_position = self.agent.get_state().position
                gt_rotation = quaternion.as_rotation_vector(self.agent.get_state().rotation)
                initial_guess = {
                    'tvec': gt_position,
                    'rvec': gt_rotation
                }
                # -- update gt local map -- #
                self.local_agent.gt_new_sim_origin = get_sim_location(gt_position,
                                                                      quaternion.from_rotation_vector(gt_rotation))
                self.local_agent.update_gt_local_map(depth_m)
                gt_local_map_pose = [self.local_agent.x_gt, self.local_agent.y_gt, self.local_agent.o_gt]
                gt_local_map, gt_local_exp_map = self.local_agent.gt_local_map, self.local_agent.gt_local_exp_map


            if use_est_map:
                est_rel_pos, est_rel_rot, keypoint_est_success = self.get_vo_relative_camera_pose(
                    prev_rgb, prev_depth, curr_rgb, action)
                rotation = R.from_rotvec(curr_rotation)

                curr_position = curr_position + rotation.apply(est_rel_pos)
                curr_rotation = curr_rotation + est_rel_rot

                self.local_agent.update_curr_pose(curr_position, curr_rotation)

                # -- update est local map -- #
                self.local_agent.est_new_sim_origin = get_sim_location(curr_position,
                                                                       quaternion.from_rotation_vector(curr_rotation))
                self.local_agent.update_est_local_map(depth_m)
                est_local_map_pose = [self.local_agent.x_est, self.local_agent.y_est, self.local_agent.o_est]
                est_local_map, est_local_exp_map = self.local_agent.est_local_map, self.local_agent.est_local_exp_map

            if visualize:
                if not os.path.exists(vis_save_dir):
                    os.makedirs(vis_save_dir)

                curr_map = vis.get_observed_colored_map(est_local_map, est_local_exp_map, self.local_agent.est_local_visit,
                                             self.local_agent.goal, gt_local_map, self.local_agent.gt_local_visit)

                out_img = vis.visualize_obs_and_map(obs['color_sensor'], obs['depth_sensor'], curr_map,
                                          est_local_map_pose, gt_local_map_pose, fig_name=f'{vis_save_dir}/local_map_{idx}',
                                          save_as_png=vis_png, save_as_video=vis_video)
                if vis_video:
                    frames.append(out_img)

                # curr_kpts0, curr_kpts1 = self.vo_module.get_matched_keypoints(prev_rgb, prev_depth, curr_rgb)
                # if not os.path.exists(f'{vis_save_dir}/vo'):
                #     os.makedirs(f'{vis_save_dir}/vo')
                # self.vo_module.viz_matching(prev_rgb, curr_rgb, curr_kpts0, curr_kpts1, path=f'{vis_save_dir}/vo/matching_{idx}')



            prev_rgb = curr_rgb
            prev_depth = curr_depth

            self.agent_steps += 1
            self.agent_length_taken += np.linalg.norm(est_rel_pos)
            if terminate_local == 1:
                break

            # print(f'{idx} step done')
        # except:
        #     print("ERROR: local navigation through error")
        #     return None, None

        if visualize and vis_video:
            video = cv2.VideoWriter(f'{vis_save_dir}/local_map.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 5, (out_img.shape[1], out_img.shape[0]))
            for frame in frames:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                video.write(frame)
            video.release()

        return

    def navigate_to_goal_point_gt(self, goal_point, bias_posi, bias_rot, visualize=False, vis_save_dir=None, vis_png=False, vis_video=True):
        rgb_list, depth_list, pano_rgb_list = [], [], []

        delta_dist, delta_rot = get_relative_location(self.local_agent.curr_pos,
                                                      self.local_agent.curr_rot,
                                                      goal_point)
        obs = self.sim.get_sensor_observations()
        self.local_agent.update_gt_local_map(obs['depth_sensor'])
        self.local_agent.update_est_local_map(obs['depth_sensor'])
        self.local_agent.set_goal(delta_dist, delta_rot)

        rgb_list.append(obs['color_sensor'])
        depth_list.append(obs['depth_sensor'])

        action_step = 0
        gt_position = self.agent.get_state().position
        gt_rotation = quaternion.as_rotation_vector(self.agent.get_state().rotation)
        # try:
        frames = []
        for idx in range(50):
            action, terminate_local = self.local_agent.navigate_local(gt=True)
            action = self.action_idx_map[action]
            action_step += 1
            obs = self.sim.step(action)
            curr_rgb = obs['color_sensor']
            curr_depth = obs["depth_sensor"]  # not normalized
            if obs['collided']:
                self.local_agent.collision = True
            depth_m = curr_depth

            rgb_list.append(curr_rgb)
            depth_list.append(curr_depth)


            # --- GT pose
            gt_position = self.agent.get_state().position - bias_posi
            gt_rotation = quaternion.as_rotation_vector(self.agent.get_state().rotation) #- bias_rot
            # -- update gt local map -- #
            self.local_agent.gt_new_sim_origin = get_sim_location(gt_position, quaternion.from_rotation_vector(gt_rotation))
            self.local_agent.update_gt_local_map(depth_m)

            if visualize:
                if not os.path.exists(vis_save_dir):
                    os.makedirs(vis_save_dir)

                gt_local_map_pose = [self.local_agent.x_gt, self.local_agent.y_gt, self.local_agent.o_gt]
                gt_local_map, gt_local_exp_map = self.local_agent.gt_local_map, self.local_agent.gt_local_exp_map
                curr_map = vis.get_observed_colored_map(gt_local_map, gt_local_exp_map, self.local_agent.est_local_visit,
                                             self.local_agent.goal, gt_local_map, self.local_agent.gt_local_visit)

                out_img = vis.visualize_obs_and_map(obs['color_sensor'], obs['depth_sensor'], curr_map,
                                          gt_local_map_pose, gt_local_map_pose, fig_name=f'{vis_save_dir}/local_map_{idx}',
                                          save_as_png=vis_png, save_as_video=vis_video)
                if vis_video:
                    frames.append(out_img)


            self.agent_steps += 1
            # self.agent_length_taken += np.linalg.norm(est_rel_pos)
            if terminate_local == 1:
                break

        if visualize and vis_video:
            video = cv2.VideoWriter(f'{vis_save_dir}/local_map.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 5, (out_img.shape[1], out_img.shape[0]))
            for frame in frames:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                video.write(frame)
            video.release()

        return rgb_list, depth_list, action_step

def main():
    import argparse
    from modules.visual_odometry.keypoint_matching import KeypointMatching
    from navigation.configs.settings_pano_navi import make_settings, make_cfg
    import habitat_sim


    parser = argparse.ArgumentParser(
        description='Image pair matching and pose evaluation with SuperGlue',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--seed', type=int, default=100)

    parser.add_argument(
        '--hfov', type=int, default=70,
        help='camera horizontal field of view')
    parser.add_argument(
        '--max_depth', type=float, default=10.,
        help='maximum depth value')
    parser.add_argument(
        '--min_depth', type=float, default=0.1,
        help='minimum depth value')

    ## local navigation configs ##
    parser.add_argument("--map_size_cm", type=int, default=1200)
    parser.add_argument("--map_resolution", type=int, default=5)

    # VO setttings
    parser.add_argument("--use_vo", type=bool, default=True)
    parser.add_argument("--model_gpu", type=str, default="0")
    parser.add_argument("--vo_width", type=int, default=320)
    parser.add_argument("--vo_height", type=int, default=240)
    parser.add_argument("--vo_hfov", type=int, default=70)

    parser.add_argument("--noisy_action", type=bool, default=False)
    parser.add_argument("--noisy_pose", type=bool, default=False)


    parser.add_argument(
        '--KM_resize', type=int, nargs='+', default=[320, 240],
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
    parser.add_argument("--sensor_height", type=float, default=1.25)


    args = parser.parse_args()
    vo_module = KeypointMatching(args)

    settings = make_settings(args, "/home/hwing/Dataset/habitat/data/scene_datasets/mp3d/1LXtFkjw3qL/1LXtFkjw3qL.glb")
    cfg = make_cfg(settings)
    sim = habitat_sim.Simulator(cfg)
    agent = sim.agents[0]

    local_navi_module = LocalNavigation(args, vo_module)

    def random_inital_agent(sim, agent, dist_from_wall=0.25):
        # initialize the agent at a random start state
        start_state = agent.get_state()
        while True:
            start_state.position = sim.pathfinder.get_random_navigable_point()
            if sim.pathfinder.distance_to_closest_obstacle(start_state.position) > dist_from_wall:
                break
        start_state.sensor_states = dict()
        return start_state

    start_state = random_inital_agent(sim, agent)
    agent.set_state(start_state)

    obs = sim.get_sensor_observations()
    position = agent.get_state().position
    rotation = agent.get_state().rotation
    rotation = quaternion.as_rotation_vector(rotation)
    local_navi_module.reset_with_curr_pose(position, rotation, sim, agent)

    while True:
        goal_position = sim.pathfinder.get_random_navigable_point()
        if np.linalg.norm(goal_position - position) < 3.0:
            break

    local_navi_module.navigate_to_goal_point(goal_position, obs)

    curr_position = agent.get_state().position
    curr_rotation = agent.get_state().rotation



def error_check():
    import argparse
    from modules.visual_odometry.keypoint_matching import KeypointMatching
    from navigation.configs.settings_pano_navi import make_settings, make_cfg
    import habitat_sim
    from tqdm import tqdm



    parser = argparse.ArgumentParser(
        description='Image pair matching and pose evaluation with SuperGlue',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    ## eval configs ##
    parser.add_argument('--seed', type=int, default=100)
    parser.add_argument('--n_env', type=int, default=50)
    parser.add_argument("--sim_gpu", type=str, default="0")
    parser.add_argument("--model_gpu", type=str, default="1")
    parser.add_argument("--gpu_list", type=str, default="8,9")


    parser.add_argument("--data_split", type=str, default="val")


    parser.add_argument(
        '--hfov', type=int, default=70,
        help='camera horizontal field of view')
    parser.add_argument(
        '--max_depth', type=float, default=10.,
        help='maximum depth value')
    parser.add_argument(
        '--min_depth', type=float, default=0.1,
        help='minimum depth value')

    ## local navigation configs ##
    parser.add_argument("--map_size_cm", type=int, default=1200)
    parser.add_argument("--map_resolution", type=int, default=5)

    # VO setttings
    parser.add_argument("--use_vo", type=bool, default=True)
    parser.add_argument("--vo_width", type=int, default=320)
    parser.add_argument("--vo_height", type=int, default=240)
    parser.add_argument("--vo_hfov", type=int, default=70)

    parser.add_argument("--noise_dir", type=str, default="/home/hwing/Projects/offline_objgoal/navigation/noise_models")
    parser.add_argument("--noisy_action", type=bool, default=False)
    parser.add_argument("--noisy_pose", type=bool, default=True)


    parser.add_argument(
        '--KM_resize', type=int, nargs='+', default=[320, 240],
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
    parser.add_argument("--sensor_height", type=float, default=1.25)


    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_list

    vo_module = KeypointMatching(args)

    dataset_dir = f'/home/hwing/Dataset/habitat/data/datasets/objectnav/mp3d/v1/{args.data_split}/content'
    dataset_list = np.sort(os.listdir(dataset_dir))

    env_name_list = os.listdir("/home/hwing/Dataset/habitat/data/scene_datasets/mp3d")
    env_name_list = [env_name for env_name in env_name_list if f'{env_name}.json.gz' in dataset_list]
    env_name_list.sort()

    out_dir_name = 'test_output/test_fmm'


    out_dir_name += f'_{args.data_split}'
    if args.noisy_action:
        out_dir_name += '_noisy_action'
    if args.noisy_pose:
        out_dir_name += '_noisy_pose'


    def random_inital_agent(sim, agent, dist_from_wall=0.25):
        # initialize the agent at a random start state
        start_state = agent.get_state()
        while True:
            start_state.position = sim.pathfinder.get_random_navigable_point()
            if sim.pathfinder.distance_to_closest_obstacle(start_state.position) > dist_from_wall:
                break
        start_state.sensor_states = dict()
        return start_state

    local_est_nav_err = []
    local_gt_nav_err = []

    for env_name in env_name_list:
        print(env_name)
        env_local_est_nav_err = []
        env_local_gt_nav_err = []


        settings = make_settings(args, f"/home/hwing/Dataset/habitat/data/scene_datasets/mp3d/{env_name}/{env_name}.glb")
        cfg = make_cfg(settings)
        sim = habitat_sim.Simulator(cfg)
        agent = sim.agents[0]

        local_navi_module = LocalNavigation(args, vo_module)
        follower = habitat_sim.GreedyGeodesicFollower(sim.pathfinder, agent)

        for idx in tqdm(range(args.n_env)):
            try:
                start_state = random_inital_agent(sim, agent)
                agent.set_state(start_state)

                obs = sim.get_sensor_observations()
                position = agent.get_state().position
                rotation = agent.get_state().rotation
                rotation = quaternion.as_rotation_vector(rotation)
                local_navi_module.reset_with_curr_pose(position, rotation, sim, agent)

                path = habitat_sim.ShortestPath()
                path.requested_start = position
                cnt = 0
                while cnt < 100:
                    goal_position = sim.pathfinder.get_random_navigable_point()
                    path.requested_end = goal_position
                    _ = sim.pathfinder.find_path(path)
                    if np.linalg.norm(goal_position - position) > 1.5 and np.linalg.norm(goal_position - position) < 3.0 and \
                        sim.pathfinder.distance_to_closest_obstacle(goal_position) > args.move_forward and \
                            path.geodesic_distance < 3.0:
                        cnt = 0
                        break
                    cnt += 1
                if not cnt < 100:
                    continue

                # if not os.path.exists(f'{out_dir_name}/{env_name}/{idx}'):
                #     os.makedirs(f'{out_dir_name}/{env_name}/{idx}')
                local_navi_module.navigate_to_goal_point(goal_position, obs, vis_save_dir=f'{out_dir_name}/{env_name}/{idx}', visualize=args.viz)

                est_position_map = np.array([local_navi_module.local_agent.x_est, local_navi_module.local_agent.y_est])
                gt_position_map = np.array([local_navi_module.local_agent.x_gt, local_navi_module.local_agent.y_gt])
                goal_position_map = np.array([local_navi_module.local_agent.goal[0], local_navi_module.local_agent.goal[1]]) * args.map_resolution

                local_est_nav_err.append(np.linalg.norm(est_position_map - goal_position_map))
                local_gt_nav_err.append(np.linalg.norm(gt_position_map - goal_position_map))

                env_local_est_nav_err.append(np.linalg.norm(est_position_map - goal_position_map))
                env_local_gt_nav_err.append(np.linalg.norm(gt_position_map - goal_position_map))

            except:
                pass


        sim.close()
        del sim

        print(f"[{env_name}] local_est_nav_err: {np.mean(env_local_est_nav_err)}")
        print(f"[{env_name}] local_gt_nav_err: {np.mean(env_local_gt_nav_err)}")

    print(f"local_est_nav_err: {np.mean(local_est_nav_err)}")
    print(f"local_gt_nav_err: {np.mean(local_gt_nav_err)}")
    print(out_dir_name)



if __name__ == "__main__":
    # main()
    error_check()
