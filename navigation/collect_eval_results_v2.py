import sys
sys.path.append('/home/hwing/Projects/OVG-Nav')

import os
import json
from utils.obj_category_info import mp3d_goal_obj_names, gibson_goal_obj_names, rednet_obj_names

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

    result_seg = [result for result in os.listdir(eval_name + '/val/results') if '.json' in result]

    for i, result in enumerate(result_seg):
        file_path = f"{eval_name}/val/results/{result}"

        # Open and load the pickled file
        try:
            with open(file_path, 'rb') as file:
                loaded_data = json.load(file)

            for key in loaded_data.keys():
                prev_success = success_results[key]['success'] * success_results[key]['count']
                prev_spl = success_results[key]['spl'] * success_results[key]['count']
                prev_dts = success_results[key]['dts'] * success_results[key]['count']
                prev_count = success_results[key]['count']
                new_success = loaded_data[key]['success'] * loaded_data[key]['count']
                new_spl = loaded_data[key]['spl'] * loaded_data[key]['count']
                new_dts = loaded_data[key]['dts'] * loaded_data[key]['count']
                new_count = loaded_data[key]['count']
                success_results[key]['count'] = prev_count + new_count
                if success_results[key]['count'] > 0:
                    success_results[key]['success'] = (prev_success + new_success) / (prev_count + new_count)
                    success_results[key]['spl'] = (prev_spl + new_spl) / (prev_count + new_count)
                    success_results[key]['dts'] = (prev_dts + new_dts) / (prev_count + new_count)
            print(f'update val_{i}  {result}')
        except:
            print(f'no file val_{i}  {result}')

    return success_results


if __name__ == '__main__':
    success_results = collect_eval_results('/disk1/hwing/Dataset/cm_graph/mp3d/val/cpu_invalid_node_v2_16-56_mp3d21_edge1v1.12_panov8_layer10_hidden512_epoch_6_goalscore_w_adjmtx_valueloss1.0_adjloss100.0_adjsimlos0.0_signloss0.0_use_cm_maxdist30.0_lr0.001',
                                           obj_type='mp3d_21')
    print(success_results)