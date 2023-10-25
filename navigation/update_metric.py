import sys
sys.path.append('/home/hwing/Projects/OVG-Nav')

import os
import json
from tqdm import tqdm
from utils.obj_category_info import mp3d_goal_obj_names, gibson_goal_obj_names, rednet_obj_names


def get_result_list(env_path):
    result_list = []
    success_result_list = [os.path.join(env_path, 'success', name, 'result.json') for name in os.listdir(os.path.join(env_path, 'success'))]
    fail_result_list = [os.path.join(env_path, 'fail', name, 'result.json') for name in os.listdir(os.path.join(env_path, 'fail'))]
    result_list.extend(success_result_list)
    result_list.extend(fail_result_list)
    result_list.sort()
    return result_list



def collect_eval_results(eval_name, obj_type='mp3d'):
    success_results = {
            'total': {'success': 0, 'spl': 0, 'dts':0, 'count': 0},
            'easy': {'success': 0, 'spl': 0, 'dts':0, 'count': 0},
            'medium': {'success': 0, 'spl': 0, 'dts':0, 'count': 0},
            'hard': {'success': 0, 'spl': 0, 'dts':0, 'count': 0},
        }

    if obj_type == 'gibson':
        goal_obj_names = gibson_goal_obj_names
    elif obj_type == 'mp3d':
        goal_obj_names = mp3d_goal_obj_names
    elif obj_type == 'mp3d_21':
        goal_obj_names = rednet_obj_names

    obj_success_results = {}
    for obj_name in goal_obj_names:
        obj_success_results[obj_name] = {'success': 0, 'spl': 0, 'dts': 0, 'count': 0}
    success_results.update(obj_success_results)

    env_name_list = [name for name in os.listdir(eval_name + '/val') if not 'result' in name]

    for env_name in env_name_list:
        result_list = get_result_list(os.path.join(eval_name, 'val', env_name))
        for result_path in tqdm(result_list):
            # Open and load the pickled file
            try:
                with open(result_path, 'rb') as file:
                    loaded_data = json.load(file)

                cur_obj_name = loaded_data['goal object']
                cur_success = loaded_data['success']
                cur_spl = loaded_data['spl']
                cur_dts = loaded_data['dts']
                cur_min_dist_to_goal = loaded_data['min_dist_to_goal_center']
                cur_shortest_path_length = loaded_data['shortest_path_length']
                cur_path_length = loaded_data['path_length']
                cur_path_level = loaded_data['path_level']


                if cur_success == 0:
                    if cur_min_dist_to_goal < 1.0:
                        cur_success = 1
                        cur_spl = cur_shortest_path_length / cur_path_length

                key_list =['total', cur_path_level, cur_obj_name]

                for key in key_list:
                    prev_success = success_results[key]['success'] * success_results[key]['count']
                    prev_spl = success_results[key]['spl'] * success_results[key]['count']
                    prev_dts = success_results[key]['dts'] * success_results[key]['count']
                    prev_count = success_results[key]['count']
                    success_results[key]['count'] = prev_count + 1
                    if success_results[key]['count'] > 0:
                        success_results[key]['success'] = (prev_success + cur_success) / (prev_count + 1)
                        success_results[key]['spl'] = (prev_spl + cur_spl) / (prev_count + 1)
                        success_results[key]['dts'] = (prev_dts + cur_dts) / (prev_count + 1)

            except:
                print(f'no file {env_name}  {result_path}')
        print(f'update {env_name}')

    return success_results


if __name__ == '__main__':
    success_results = collect_eval_results('/disk1/hwing/Dataset/cm_graph/mp3d/val/0918_vonomatch_voiter5_noiseupdate_30_cpu_invalid_node_v2_16-56_mp3d21_edge1v1.12_panov8_layer10_hidden512_epoch_6_goalscore_w_adjmtx_valueloss1.0_adjloss100.0_adjsimlos0.0_signloss0.0_use_cm_maxdist30.0_lr0.001',
                                           obj_type='mp3d_21')
    print(success_results)