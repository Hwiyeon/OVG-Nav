#! /bin/bash

cd ../

#python navigation/eval_pano.py --gpu_list '1' --run_type 'val' --data_split 0  --data_split_max=11 &
#python navigation/eval_pano.py --gpu_list '1' --run_type 'val' --data_split 1  --data_split_max=11 &
#python navigation/eval_pano.py --gpu_list '1' --run_type 'val' --data_split 2  --data_split_max=11 &
#python navigation/eval_pano.py --gpu_list '2' --run_type 'val' --data_split 3  --data_split_max=11 &
#python navigation/eval_pano.py --gpu_list '2' --run_type 'val' --data_split 4  --data_split_max=11 &
#python navigation/eval_pano.py --gpu_list '2' --run_type 'val' --data_split 5  --data_split_max=11 &
#python navigation/eval_pano.py --gpu_list '3' --run_type 'val' --data_split 6  --data_split_max=11 &
#python navigation/eval_pano.py --gpu_list '3' --run_type 'val' --data_split 7  --data_split_max=11 &
#python navigation/eval_pano.py --gpu_list '3' --run_type 'val' --data_split 8  --data_split_max=11 &
#python navigation/eval_pano.py --gpu_list '4' --run_type 'val' --data_split 9  --data_split_max=11 &
#python navigation/eval_pano.py --gpu_list '4' --run_type 'val' --data_split 10  --data_split_max=11 &




python navigation/eval_pano.py --gpu_list '1' --run_type 'val' --data_split 0  --data_split_max=11 --in_data_split 0 --in_data_split_max 2 &
python navigation/eval_pano.py --gpu_list '1' --run_type 'val' --data_split 0  --data_split_max=11 --in_data_split 1 --in_data_split_max 2 &
python navigation/eval_pano.py --gpu_list '1' --run_type 'val' --data_split 1  --data_split_max=11 --in_data_split 0 --in_data_split_max 2 &
python navigation/eval_pano.py --gpu_list '1' --run_type 'val' --data_split 1  --data_split_max=11 --in_data_split 1 --in_data_split_max 2 &
python navigation/eval_pano.py --gpu_list '2' --run_type 'val' --data_split 2  --data_split_max=11 --in_data_split 0 --in_data_split_max 2 &
python navigation/eval_pano.py --gpu_list '2' --run_type 'val' --data_split 2  --data_split_max=11 --in_data_split 1 --in_data_split_max 2 &
python navigation/eval_pano.py --gpu_list '6' --run_type 'val' --data_split 3  --data_split_max=11 --in_data_split 0 --in_data_split_max 2 &
python navigation/eval_pano.py --gpu_list '6' --run_type 'val' --data_split 3  --data_split_max=11 --in_data_split 1 --in_data_split_max 2 &
python navigation/eval_pano.py --gpu_list '3' --run_type 'val' --data_split 4  --data_split_max=11 --in_data_split 0 --in_data_split_max 2 &
python navigation/eval_pano.py --gpu_list '3' --run_type 'val' --data_split 4  --data_split_max=11 --in_data_split 1 --in_data_split_max 2 &
python navigation/eval_pano.py --gpu_list '3' --run_type 'val' --data_split 5  --data_split_max=11 --in_data_split 0 --in_data_split_max 2 &
python navigation/eval_pano.py --gpu_list '3' --run_type 'val' --data_split 5  --data_split_max=11 --in_data_split 1 --in_data_split_max 2 &
python navigation/eval_pano.py --gpu_list '4' --run_type 'val' --data_split 6  --data_split_max=11 --in_data_split 0 --in_data_split_max 2 &
python navigation/eval_pano.py --gpu_list '4' --run_type 'val' --data_split 6  --data_split_max=11 --in_data_split 1 --in_data_split_max 2 &
python navigation/eval_pano.py --gpu_list '5' --run_type 'val' --data_split 7  --data_split_max=11 --in_data_split 0 --in_data_split_max 2 &
python navigation/eval_pano.py --gpu_list '5' --run_type 'val' --data_split 7  --data_split_max=11 --in_data_split 1 --in_data_split_max 2 &
python navigation/eval_pano.py --gpu_list '4' --run_type 'val' --data_split 8  --data_split_max=11 --in_data_split 0 --in_data_split_max 2 &
python navigation/eval_pano.py --gpu_list '4' --run_type 'val' --data_split 8  --data_split_max=11 --in_data_split 1 --in_data_split_max 2 &
python navigation/eval_pano.py --gpu_list '5' --run_type 'val' --data_split 9  --data_split_max=11 --in_data_split 0 --in_data_split_max 2 &
python navigation/eval_pano.py --gpu_list '5' --run_type 'val' --data_split 9  --data_split_max=11 --in_data_split 1 --in_data_split_max 2 &
python navigation/eval_pano.py --gpu_list '6' --run_type 'val' --data_split 10  --data_split_max=11 --in_data_split 0 --in_data_split_max 2 &
python navigation/eval_pano.py --gpu_list '6' --run_type 'val' --data_split 10  --data_split_max=11 --in_data_split 1 --in_data_split_max 2 &







































