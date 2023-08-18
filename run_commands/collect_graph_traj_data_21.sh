#! /bin/bash

cd ../

#python navigation/data_collect_21.py --gpu_list '1' --run_type 'val' --data_split_max 1 &
python navigation/data_collect_21.py --gpu_list '1' --run_type 'train' --data_split 0 &
python navigation/data_collect_21.py --gpu_list '1' --run_type 'train' --data_split 1 &
python navigation/data_collect_21.py --gpu_list '1' --run_type 'train' --data_split 2 &
python navigation/data_collect_21.py --gpu_list '2' --run_type 'train' --data_split 3 &
python navigation/data_collect_21.py --gpu_list '2' --run_type 'train' --data_split 4 &
python navigation/data_collect_21.py --gpu_list '2' --run_type 'train' --data_split 5 &
python navigation/data_collect_21.py --gpu_list '2' --run_type 'train' --data_split 6 &
python navigation/data_collect_21.py --gpu_list '3' --run_type 'train' --data_split 7 &
python navigation/data_collect_21.py --gpu_list '3' --run_type 'train' --data_split 8 &
python navigation/data_collect_21.py --gpu_list '3' --run_type 'train' --data_split 9 &
python navigation/data_collect_21.py --gpu_list '3' --run_type 'train' --data_split 10 &
python navigation/data_collect_21.py --gpu_list '4' --run_type 'train' --data_split 11 &
python navigation/data_collect_21.py --gpu_list '4' --run_type 'train' --data_split 12 &
python navigation/data_collect_21.py --gpu_list '4' --run_type 'train' --data_split 13 &
python navigation/data_collect_21.py --gpu_list '3' --run_type 'train' --data_split 14 &
python navigation/data_collect_21.py --gpu_list '3' --run_type 'train' --data_split 15 &
python navigation/data_collect_21.py --gpu_list '3' --run_type 'train' --data_split 16 &
python navigation/data_collect_21.py --gpu_list '4' --run_type 'train' --data_split 17 &
python navigation/data_collect_21.py --gpu_list '4' --run_type 'train' --data_split 18 &
python navigation/data_collect_21.py --gpu_list '4' --run_type 'train' --data_split 19 &
#python navigation/data_collect_21.py --gpu_list '8' --run_type 'train' --data_split 20 &
#python navigation/data_collect_21.py --gpu_list '8' --run_type 'train' --data_split 21 &
#python navigation/data_collect_21.py --gpu_list '8' --run_type 'train' --data_split 22 &
#python navigation/data_collect_21.py --gpu_list '9' --run_type 'train' --data_split 23 &
#python navigation/data_collect_21.py --gpu_list '9' --run_type 'train' --data_split 24 &
#python navigation/data_collect_21.py --gpu_list '9' --run_type 'train' --data_split 25 &
#python navigation/data_collect.py --gpu_list '6' --run_type 'train' --data_split 26 &
#python navigation/data_collect.py --gpu_list '7' --run_type 'train' --data_split 27 &
#python navigation/data_collect.py --gpu_list '7' --run_type 'train' --data_split 28 &
#python navigation/data_collect.py --gpu_list '7' --run_type 'train' --data_split 29 &
#python navigation/data_collect.py --gpu_list '7' --run_type 'train' --data_split 30 &

#python navigation/data_collect_21.py --gpu_list '5' --run_type 'train' --data_split 3 &
#python navigation/data_collect_21.py --gpu_list '5' --run_type 'train' --data_split 4 &
#python navigation/data_collect_21.py --gpu_list '5' --run_type 'train' --data_split 5 &
#python navigation/data_collect_21.py --gpu_list '5' --run_type 'train' --data_split 6 &











































