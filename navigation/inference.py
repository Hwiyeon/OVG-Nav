import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "8, 9"

import argparse
import numpy as np
import time
import json
import gzip

from navigation.configs.settings_pano_navi import make_settings
import inference_runner as runner


parser = argparse.ArgumentParser()

## eval configs ##
parser.add_argument("--gpu_list", type=str, default="9")
parser.add_argument("--model_gpu", type=str, default="1")
parser.add_argument("--sim_gpu", type=str, default="0")
parser.add_argument("--data", type=str, default="mp3d")
parser.add_argument("--n_for_env", type=int, default=2000)
parser.add_argument("--max_step", type=int, default=500)
parser.add_argument("--run_type", type=str, default="train")
parser.add_argument("--data_split", type=int, default=2)
parser.add_argument("--data_split_max", type=int, default=6)
parser.add_argument("--save_dir", type=str, default="./logs/0516_test_edge1.0_height_0.88_train")
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--scene", type=str, default="/home/hwing/Dataset/habitat/data/scene_datasets/mp3d")
parser.add_argument("--floorplan_data_dir", type=str, default='/home/hwing/Dataset/habitat/data/floorplans')


## observation configs ##
parser.add_argument("--front_width", type=int, default=320)
parser.add_argument("--front_height", type=int, default=240)
parser.add_argument("--front_hfov", type=int, default=70)

parser.add_argument("--add_panoramic_sensor", type=bool, default=True)
parser.add_argument("--panoramic_turn_angle", type=int, default=90)
parser.add_argument("--width", type=int, default=128)
parser.add_argument("--height", type=int, default=128)
parser.add_argument("--hfov", type=int, default=90)
parser.add_argument("--pano_width", type=int, default=512)
parser.add_argument("--pano_height", type=int, default=128)
parser.add_argument("--sensor_height", type=float, default=0.88)

parser.add_argument("--semantic_sensor", type=bool, default=False)


## agent configs ##
parser.add_argument("--max_frames", type=int, default=500)
parser.add_argument("--sensing_range", type=float, default=10.0)
parser.add_argument("--move_forward", type=float, default=0.25)
parser.add_argument("--edge_range", type=float, default=1.0)
parser.add_argument("--act_rot", type=int, default=10)
parser.add_argument("--cand_rot", type=int, default=30)

## noise configs ##
parser.add_argument('--noisy_rgb', type=bool, default=False, help='use Gaussian noise on RGB')
parser.add_argument('--noisy_rgb_multiplier', type=float, default=0.1, help='use Gaussian noise on RGB')
parser.add_argument('--noisy_depth', type=bool, default=False, help='use RedwoodDepthNoiseModel')
parser.add_argument('--noisy_depth_multiplier', type=float, default=5., help='use RedwoodDepthNoiseModel noise multiplier')
parser.add_argument("--noise_dir", type=str, default="navigation/noise_models")
parser.add_argument('--noisy_action', type=bool, default=False, help='')
parser.add_argument('--noisy_pose', type=bool, default=False, help='')

## local navigation configs ##
parser.add_argument("--map_size_cm", type=int, default=1200)
parser.add_argument("--map_resolution", type=int, default=5)


## model configs ##
parser.add_argument("--detection_model", type=str, default="/home/hwing/Projects/offline_objgoal/modules/detector")
parser.add_argument("--free_space_model", type=str, default="/home/hwing/Projects/offline_objgoal/modules/free_space_model/ckpts/split_lr0.001_0227_range_2.0/best_model_1.pth")
parser.add_argument("--CLIP_model", type=str, default="ViT-B/32")
parser.add_argument("--COMET_model", type=str, default="/home/hwing/Projects/offline_objgoal/modules/comet_relation/comet-atomic_2020_BART")

## VO model configs ##
parser.add_argument("--use_vo", type=bool, default=False)
parser.add_argument('--max_depth', type=float, default=10., help='maximum depth value')
parser.add_argument('--min_depth', type=float, default=0.1, help='minimum depth value')
parser.add_argument('--KM_resize', type=int, nargs='+', default=[320, 240],
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





# dataset_dir = '/home/hwing/Projects/obj_navi/data/datasets/objectnav/mp3d/v1/train/content'
# dataset_list = np.sort(os.listdir(dataset_dir))

def get_data_list(args):
    dataset_dir = f'/home/hwing/Dataset/habitat/data/datasets/objectnav/mp3d/v1/{args.run_type}/content'
    dataset_list = np.sort([os.path.join(dataset_dir, dataset) for dataset in os.listdir(dataset_dir)])
    dataset_env_list = np.sort(os.listdir(dataset_dir))
    env_list = os.listdir(args.scene)
    env_list = [data for data in env_list if f'{data}.json.gz' in dataset_env_list]
    env_list = np.sort(env_list)


    if args.run_type == 'train':
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


obj_names = ["chair", "table", "picture","cabinet", "cushion",
             "sofa", "bed", "chest_of_drawers", "plant", "sink",
             "toilet", "stool", "towel", "tv_monitor", "shower",
             "bathhub", "counter", "fireplace", "seating","gym_equipment", "clothes"]

# goal_obj_names = ["bed", "chair", "plant", "sofa", "toilet", "tv_monitor"]

goal_obj_names = ['chair',         # 0
                  'sofa',         # 1
                  'plant',  # 2
                  'bed',           # 3
                  'toilet',        # 4
                  'tv_monitor'             # 5
                  ]


def main(env_list, dataset_list):

    for i, scene_name in enumerate(env_list):
        env_start_time = time.time()
        settings = make_settings(args, args.scene+"/{}/{}.glb".format(scene_name, scene_name))


        with gzip.open(f'{dataset_list[i]}', 'r') as f:
            dataset = f.read()
        dataset = json.loads(dataset.decode('utf-8'))

        demo_runner = runner.Runner(args, settings, det_COI, dataset, data_type=args.run_type)
        demo_runner.get_data(i + 1, len(env_list))


        print(f"[{i+1}/{len(env_list)}] Done")




if __name__=='__main__':
    env_list, dataset_list = get_data_list(args)
    main(env_list, dataset_list)
