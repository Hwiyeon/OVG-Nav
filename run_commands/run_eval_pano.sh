#! /bin/bash

cd ../

python navigation/eval_pano.py --gpu_list '7' --run_type 'val' --data_split 0  --data_split_max=11 &
python navigation/eval_pano.py --gpu_list '7' --run_type 'val' --data_split 1  --data_split_max=11 &
python navigation/eval_pano.py --gpu_list '7' --run_type 'val' --data_split 2  --data_split_max=11 &
python navigation/eval_pano.py --gpu_list '7' --run_type 'val' --data_split 3  --data_split_max=11 &
python navigation/eval_pano.py --gpu_list '8' --run_type 'val' --data_split 4  --data_split_max=11 &
python navigation/eval_pano.py --gpu_list '8' --run_type 'val' --data_split 5  --data_split_max=11 &
python navigation/eval_pano.py --gpu_list '8' --run_type 'val' --data_split 6  --data_split_max=11 &
python navigation/eval_pano.py --gpu_list '9' --run_type 'val' --data_split 7  --data_split_max=11 &
#python navigation/eval.py --gpu_list '2' --run_type 'val' --data_split 8  --data_split_max=11 &
python navigation/eval_pano.py --gpu_list '9' --run_type 'val' --data_split 9  --data_split_max=11 &
python navigation/eval_pano.py --gpu_list '9' --run_type 'val' --data_split 10  --data_split_max=11 &










































