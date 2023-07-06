#! /bin/bash

cd ../

#python navigation/data_collect.py --gpu_list '8, 9' --run_type 'val' &
python navigation/data_collect_randomtraj.py --gpu_list '0, 1' --run_type 'train' --data_split 0 &
python navigation/data_collect_randomtraj.py --gpu_list '0, 1' --run_type 'train' --data_split 1 &
python navigation/data_collect_randomtraj.py --gpu_list '0, 1' --run_type 'train' --data_split 2 &
python navigation/data_collect_randomtraj.py --gpu_list '0, 1' --run_type 'train' --data_split 3 &
python navigation/data_collect_randomtraj.py --gpu_list '2, 3' --run_type 'train' --data_split 4 &
python navigation/data_collect_randomtraj.py --gpu_list '2, 3' --run_type 'train' --data_split 5 &
python navigation/data_collect_randomtraj.py --gpu_list '2, 3' --run_type 'train' --data_split 6 &
python navigation/data_collect_randomtraj.py --gpu_list '2, 3' --run_type 'train' --data_split 7 &
python navigation/data_collect_randomtraj.py --gpu_list '2, 3' --run_type 'train' --data_split 8 &
python navigation/data_collect_randomtraj.py --gpu_list '4, 5' --run_type 'train' --data_split 9 &
python navigation/data_collect_randomtraj.py --gpu_list '4, 5' --run_type 'train' --data_split 10 &
python navigation/data_collect_randomtraj.py --gpu_list '4, 5' --run_type 'train' --data_split 11 &
python navigation/data_collect_randomtraj.py --gpu_list '4, 5' --run_type 'train' --data_split 12 &
python navigation/data_collect_randomtraj.py --gpu_list '4, 5' --run_type 'train' --data_split 13 &
python navigation/data_collect_randomtraj.py --gpu_list '6, 7' --run_type 'train' --data_split 14 &
python navigation/data_collect_randomtraj.py --gpu_list '6, 7' --run_type 'train' --data_split 15 &
python navigation/data_collect_randomtraj.py --gpu_list '6, 7' --run_type 'train' --data_split 16 &
python navigation/data_collect_randomtraj.py --gpu_list '6, 7' --run_type 'train' --data_split 17 &
python navigation/data_collect_randomtraj.py --gpu_list '6, 7' --run_type 'train' --data_split 18 &
python navigation/data_collect_randomtraj.py --gpu_list '8, 9' --run_type 'train' --data_split 19 &
python navigation/data_collect_randomtraj.py --gpu_list '8, 9' --run_type 'train' --data_split 20 &
python navigation/data_collect_randomtraj.py --gpu_list '8, 9' --run_type 'train' --data_split 21 &












































