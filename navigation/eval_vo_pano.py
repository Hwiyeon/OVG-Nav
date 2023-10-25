import bz2
import os
os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["CUDA_VISIBLE_DEVICES"] = "8, 9"

import sys
sys.path.append('/home/hwing/Projects/OVG-Nav')

import argparse


parser = argparse.ArgumentParser()

## eval configs ##
parser.add_argument("--gpu_list", type=str, default="0")
parser.add_argument("--model_gpu", type=str, default="0")
parser.add_argument("--sim_gpu", type=str, default="0")
# parser.add_argument("--data", type=str, default="gibosn")
parser.add_argument("--n_for_env", type=int, default=2000)
parser.add_argument("--debug_frame", type=bool, default=False)
parser.add_argument("--max_step", type=int, default=500)
parser.add_argument("--dataset", type=str, default='mp3d')
parser.add_argument("--run_type", type=str, default="val")
parser.add_argument("--data_split", type=int, default=10)
parser.add_argument("--data_split_max", type=int, default=11)
parser.add_argument("--in_data_split", type=int, default=0)
parser.add_argument("--in_data_split_max", type=int, default=1)
# parser.add_argument("--save_dir", type=str, default="/disk1/hwing/Dataset/cm_graph/{}/{}/test/test9")
parser.add_argument("--save_dir", type=str, default="/disk1/hwing/Dataset/cm_graph/{}/{}/0920_vonomatch_voiter5_noiseupdate_30_cpu_invalid_node_v2_16-56_mp3d21_edge1v1.12_panov8_layer10_hidden512_epoch_6_goalscore_w_adjmtx_valueloss1.0_adjloss100.0_adjsimlos0.0_signloss0.0_use_cm_maxdist30.0_lr0.001")
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--scene", type=str, default="/home/hwing/Dataset/habitat/data/scene_datasets/{}")
parser.add_argument("--dataset_dir", type=str, default="/home/hwing/Dataset/habitat/data/datasets/objectnav/{}/{}/{}/content")
parser.add_argument("--floorplan_data_dir", type=str, default='/home/hwing/Dataset/habitat/data/floorplans')
parser.add_argument("--vis_floorplan", type=bool, default=True)
parser.add_argument("--use_oracle", type=bool, default=False)
parser.add_argument("--cm_type", type=str, default="comet")
parser.add_argument('--goal_cat', type=str, default='mp3d_21')


## observation configs ##
parser.add_argument("--front_width", type=int, default=640)
parser.add_argument("--front_height", type=int, default=480)
parser.add_argument("--front_hfov", type=int, default=70)

parser.add_argument("--add_panoramic_sensor", type=bool, default=True)
parser.add_argument("--panoramic_turn_angle", type=int, default=90)
parser.add_argument("--width", type=int, default=256)
parser.add_argument("--height", type=int, default=256)
parser.add_argument("--hfov", type=int, default=90)
parser.add_argument("--pano_width", type=int, default=1024)
parser.add_argument("--pano_height", type=int, default=256)
parser.add_argument("--sensor_height", type=float, default=0.88)
parser.add_argument("--goal_obs_count", type=int, default=3)

parser.add_argument("--semantic_sensor", type=bool, default=False)


## agent configs ##
parser.add_argument("--max_frames", type=int, default=500)
parser.add_argument("--sensing_range", type=float, default=5.0)
parser.add_argument("--move_forward", type=float, default=0.25)
parser.add_argument("--edge_range", type=float, default=1.0)
parser.add_argument("--last_mile_range", type=float, default=5.0)
parser.add_argument("--act_rot", type=int, default=30)
parser.add_argument("--cand_rot", type=int, default=30)
parser.add_argument("--success_dist", type=float, default=1.0)

## noise configs ##
parser.add_argument('--noisy_rgb', type=bool, default=False, help='use Gaussian noise on RGB')
parser.add_argument('--noisy_rgb_multiplier', type=float, default=0.1, help='use Gaussian noise on RGB')
parser.add_argument('--noisy_depth', type=bool, default=True, help='use RedwoodDepthNoiseModel')
parser.add_argument('--noisy_depth_multiplier', type=float, default=5., help='use RedwoodDepthNoiseModel noise multiplier')
parser.add_argument("--noise_dir", type=str, default="navigation/noise_models")
parser.add_argument('--noisy_action', type=bool, default=True, help='')
parser.add_argument('--noisy_action_multiplier', type=float, default=1.0, help='')

## local navigation configs ##
parser.add_argument("--map_size_cm", type=int, default=1200)
parser.add_argument("--map_resolution", type=int, default=5)


## model configs ##
parser.add_argument("--detection_model", type=str, default="/home/hwing/Projects/OVG-Nav/modules/detector")
parser.add_argument("--segmentation_model", type=str, default="/home/hwing/Projects/OVG-Nav/modules/detector/rednet_mp3d.pth")
parser.add_argument("--free_space_model", type=str, default="/home/hwing/Projects/OVG-Nav/modules/free_space_model/ckpts/split_lr0.001_0227_range_1.0/best_model_1.pth")
parser.add_argument("--CLIP_model", type=str, default="/home/hwing/Projects/OVG-Nav/modules/comet_relation/clip-vit-base-patch32/ViT-B-32.pt")
parser.add_argument("--COMET_model", type=str, default="/home/hwing/Projects/OVG-Nav/modules/comet_relation/comet-atomic_2020_BART")

# parser.add_argument("--value_model", type=str, default='/data1/hwing/Projects/offline_objgoal/goal_dist_pred/logs/cm_0610/0610_v2_1_use_cm_maxdist30.0_lr0.001/model_25.pth')
# parser.add_argument("--value_model", type=str, default='/data1/hwing/Projects/offline_objgoal/goal_dist_pred/logs/cm_0616/0616_combv2_modelv2_1_use_cm_maxdist30.0_lr0.0001/model_20.pth')
parser.add_argument("--value_model", type=str, default='/home/hwing/Projects/OVG-Nav/goal_dist_pred/models/16-56_mp3d21_edge1v1.12_panov8_layer10_hidden512_epoch_6_goalscore_w_adjmtx_valueloss1.0_adjloss100.0_adjsimlos0.0_signloss0.0_use_cm_maxdist30.0_lr0.001/model.pth')
parser.add_argument('--vis_feat_dim', default=512, type=int)
parser.add_argument('--gcn_layers', default=10, type=int)
parser.add_argument('--use_cm_score', default=True, type=bool)


## VO model configs ##
parser.add_argument("--use_vo", type=bool, default=True)
parser.add_argument('--max_depth', type=float, default=5., help='maximum depth value')
parser.add_argument('--min_depth', type=float, default=0.1, help='minimum depth value')
parser.add_argument('--KM_resize', type=int, nargs='+', default=[640, 480],
        help='Resize the input image before running inference. If two numbers, '
             'resize to the exact dimensions, if one number, resize the max '
             'dimension, if -1, do not resize')
parser.add_argument('--KM_superglue', choices={'indoor', 'outdoor'}, default='indoor', help='SuperGlue weights')
parser.add_argument('--KM_max_keypoints', type=int, default=1024, help='Maximum number of keypoints detected by Superpoint' ' (\'-1\' keeps all keypoints)')
parser.add_argument('--KM_keypoint_threshold', type=float, default=0.005, help='SuperPoint keypoint detector confidence threshold')
parser.add_argument('--KM_nms_radius', type=int, default=4, help='SuperPoint Non Maximum Suppression (NMS) radius' ' (Must be positive)')
parser.add_argument('--KM_sinkhorn_iterations', type=int, default=20, help='Number of Sinkhorn iterations performed by SuperGlue')
parser.add_argument('--KM_match_threshold', type=float, default=0.2, help='SuperGlue match threshold')


args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_list
import numpy as np
import time
import json
import gzip
import _pickle as cPickle

from navigation.configs.settings_pano_navi import make_settings
import eval_runner_vo_pano as runner
from utils.obj_category_info import gibson_goal_obj_names, mp3d_goal_obj_names, rednet_obj_names



args.save_dir = args.save_dir.format(args.dataset, args.run_type)

if args.dataset =='mp3d' or args.dataset =='gibson':
    args.scene = args.scene.format(args.dataset)
elif args.dataset =='hm3d':
    args.scene = args.scene.format(f'{args.dataset}/v1/{args.run_type}')


if args.dataset =='mp3d':
    args.dataset_dir = args.dataset_dir.format(args.dataset, 'v1', args.run_type)
elif args.dataset =='gibson':
    args.dataset_dir = args.dataset_dir.format(args.dataset, 'v1.1', args.run_type)
elif args.dataset =='hm3d':
    args.dataset_dir = args.dataset_dir.format(args.dataset, 'v1', args.run_type)


args.KM_resize = [int(args.front_width), int(args.front_height)]



def get_data_list(args):
    # dataset_dir = f'/home/hwing/Dataset/habitat/data/datasets/objectnav/mp3d/v1/{args.run_type}/content'
    dataset_dir = args.dataset_dir
    if args.dataset == 'mp3d' or args.dataset == 'hm3d':
        dataset_list = np.sort([os.path.join(dataset_dir, dataset) for dataset in os.listdir(dataset_dir)])
    elif args.dataset == 'gibson':
        dataset_list = np.sort([os.path.join(dataset_dir, dataset) for dataset in os.listdir(dataset_dir) if 'episodes' in dataset])
    dataset_env_list = np.sort(os.listdir(dataset_dir))
    env_list = os.listdir(args.scene)

    if args.dataset == 'mp3d' or args.dataset == 'gibson':
        env_list = [data for data in env_list if f'{data}.json.gz' in dataset_env_list]
    elif args.dataset == 'hm3d':
        env_list = [data for data in env_list if '{}.json.gz'.format(data.split('-')[-1]) in dataset_env_list]

    env_list = np.sort(env_list)


    # if args.run_type == 'train':
    if args.data_split == args.data_split_max-1:
        dataset_list = dataset_list[int(args.data_split * len(dataset_list) / args.data_split_max):]
        env_list = env_list[int(args.data_split * len(env_list) / args.data_split_max):]
    else:
        dataset_list = dataset_list[int(args.data_split*len(dataset_list)/args.data_split_max):int((args.data_split+1)*len(dataset_list)/args.data_split_max)]
        env_list = env_list[int(args.data_split*len(env_list)/args.data_split_max):int((args.data_split+1)*len(env_list)/args.data_split_max)]
    print(env_list)

    return env_list, dataset_list

##
# data_list = ['B6ByNegPMKs']



# COI = [3,  # Chair,
#        4,  #Table,
#        6,  #Picture,
#        7,  #Cabinet,
#        8,  #Cushion,
#        10,  #Sofa,
#        11,  #Bed,
#        13,  #Chest of Drawers,9
#        14,  #Plant,
#        15,  #Sink,
#        18,  # Toilet,
#        19,  # Stool,
#        20,  #Towel,
#        22,  # TV Monitor
#        23,  #Shower,
#        25,  # Bathhub,
#        26,  #Counter,
#        27,  # Fireplace,
#        34,  # Seating,
#        33,  #Gym Equipment,
#        38,  #Clothes,
#        ]

det_COI = [
        56,  # chair
        57,  # couch
        58,  # potted plant
        59,  # bed
        61,  # toilet
        62,  # tv
        60,  # dining table
        63,  # laptop
        68,  # microwave
        69,  # oven
        71,  # sink
        72,  # refrigerator
        74,  # clock
        75,  # vase
    ]

#
# obj_names = ["chair", "table", "picture","cabinet", "cushion",
#              "sofa", "bed", "chest_of_drawers", "plant", "sink",
#              "toilet", "stool", "towel", "tv_monitor", "shower",
#              "bathhub", "counter", "fireplace", "seating","gym_equipment", "clothes"]

# goal_obj_names = ["bed", "chair", "plant", "sofa", "toilet", "tv_monitor"]

# goal_obj_names = ['chair',         # 0
#                   'sofa',         # 1
#                   'plant',  # 2
#                   'bed',           # 3
#                   'toilet',        # 4
#                   'tv_monitor'             # 5
#                   ]
#

def main(env_list, dataset_list):
    success_results = {
        'total': {'success': 0, 'spl': 0, 'dts': 0, 'count': 0},
        'easy': {'success': 0, 'spl': 0, 'dts': 0, 'count': 0},
        'medium': {'success': 0, 'spl': 0, 'dts': 0, 'count': 0},
        'hard': {'success': 0, 'spl': 0, 'dts': 0, 'count': 0},
    }

    # env_list, dataset_list = env_list[1:], dataset_list[1:]

    if args.dataset == 'mp3d' or args.dataset == 'hm3d':
        goal_obj_names = mp3d_goal_obj_names
    elif args.dataset == 'gibson':
        goal_obj_names = gibson_goal_obj_names
        dataset_info_file = args.dataset_dir.replace('content', f'{args.run_type}_info.pbz2')
        with bz2.BZ2File(dataset_info_file, 'rb') as f:
            dataset_info = cPickle.load(f)


    if args.goal_cat == 'mp3d_21':
        goal_obj_names = rednet_obj_names

    obj_success_results = {}
    for obj_name in goal_obj_names:
        obj_success_results[obj_name] = {'success': 0, 'spl': 0, 'dts': 0, 'count': 0}

    # obj_success_results = {
    #     'chair': {'success': 0, 'spl': 0, 'count': 0},
    #     'sofa': {'success': 0, 'spl': 0, 'count': 0},
    #     'plant': {'success': 0, 'spl': 0, 'count': 0},
    #     'bed': {'success': 0, 'spl': 0, 'count': 0},
    #     'toilet': {'success': 0, 'spl': 0, 'count': 0},
    #     'tv_monitor': {'success': 0, 'spl': 0, 'count': 0},
    # }
    for i, scene_name in enumerate(env_list):
        env_start_time = time.time()
        if args.dataset == 'mp3d':
            settings = make_settings(args, args.scene+"/{}/{}.glb".format(scene_name, scene_name))
        elif args.dataset == 'gibson':
            settings = make_settings(args, args.scene + "/{}".format(scene_name))
        elif args.dataset == 'hm3d':
            settings = make_settings(args, args.scene + "/{}/{}.basis.glb".format(scene_name, scene_name.split('-')[-1]))


        with gzip.open(f'{dataset_list[i]}', 'r') as f:
            dataset = f.read()
        dataset = json.loads(dataset.decode('utf-8'))

        demo_runner = runner.Runner(args, settings, det_COI, dataset, data_type=args.run_type)
        if args.dataset == 'gibson':
            demo_runner.dataset_info = dataset_info[scene_name.split('.')[0]]
        env_success_results, env_obj_success_results = demo_runner.get_data(i + 1, len(env_list))

        if env_success_results is not None:
            for key in success_results.keys():
                prev_success = success_results[key]['success'] * success_results[key]['count']
                prev_spl = success_results[key]['spl'] * success_results[key]['count']
                prev_dts = success_results[key]['dts'] * success_results[key]['count']
                prev_count = success_results[key]['count']
                new_success = env_success_results[key]['success'] * env_success_results[key]['count']
                new_spl = env_success_results[key]['spl'] * env_success_results[key]['count']
                new_dts = env_success_results[key]['dts'] * env_success_results[key]['count']
                new_count = env_success_results[key]['count']
                success_results[key]['count'] = prev_count + new_count
                if success_results[key]['count'] > 0:
                    success_results[key]['success'] = (prev_success + new_success) / (prev_count + new_count)
                    success_results[key]['spl'] = (prev_spl + new_spl) / (prev_count + new_count)
                    success_results[key]['dts'] = (prev_dts + new_dts) / (prev_count + new_count)


            for key in obj_success_results.keys():
                prev_success = obj_success_results[key]['success'] * obj_success_results[key]['count']
                prev_spl = obj_success_results[key]['spl'] * obj_success_results[key]['count']
                prev_dts = obj_success_results[key]['dts'] * obj_success_results[key]['count']
                prev_count = obj_success_results[key]['count']
                new_success = env_obj_success_results[key]['success'] * env_obj_success_results[key]['count']
                new_spl = env_obj_success_results[key]['spl'] * env_obj_success_results[key]['count']
                new_dts = env_obj_success_results[key]['dts'] * env_obj_success_results[key]['count']
                new_count = env_obj_success_results[key]['count']
                obj_success_results[key]['count'] = prev_count + new_count
                if obj_success_results[key]['count'] > 0:
                    obj_success_results[key]['success'] = (prev_success + new_success) / (prev_count + new_count)
                    obj_success_results[key]['spl'] = (prev_spl + new_spl) / (prev_count + new_count)
                    obj_success_results[key]['dts'] = (prev_dts + new_dts) / (prev_count + new_count)


            print(f"[{i+1}/{len(env_list)}] Done")
            print(
                f"     Total - success: {success_results['total']['success']}, spl: {success_results['total']['spl']}, dts: {success_results['total']['dts']}, count: {success_results['total']['count']} \n"
                f"     Easy - success: {success_results['easy']['success']}, spl: {success_results['easy']['spl']}, dts: {success_results['easy']['dts']}, count: {success_results['easy']['count']} \n"
                f"     Medium - success: {success_results['medium']['success']}, spl: {success_results['medium']['spl']}, dts: {success_results['medium']['dts']}, count: {success_results['medium']['count']} \n"
                f"     Hard - success: {success_results['hard']['success']}, spl: {success_results['hard']['spl']}, dts: {success_results['hard']['dts']}, count: {success_results['hard']['count']} \n")
            for obj_name in goal_obj_names:
                print(
                    f"     {obj_name} - success: {obj_success_results[obj_name]['success']}, spl: {obj_success_results[obj_name]['spl']}, dts: {obj_success_results[obj_name]['dts']}, count: {obj_success_results[obj_name]['count']}")

    with open(f'{args.save_dir}/{args.run_type}_{args.data_split}_result.json','wb') as f:
        success_results.update(obj_success_results)
        json_data = json.dumps(success_results).encode('utf-8')
        f.write(json_data)



if __name__=='__main__':
    env_list, dataset_list = get_data_list(args)
    main(env_list, dataset_list)
