#!/bin/bash

for seed in {42..91}; do
    ~/.conda/envs/deeplearning/bin/python source.py --real_name training --gen_name generated_consecutive --cuda 2 --dataset_seed $seed &
    ~/.conda/envs/deeplearning/bin/python source.py --real_name training --gen_name add_weather_consecutive --cuda 2 --dataset_seed $seed &
    ~/.conda/envs/deeplearning/bin/python source.py --real_name training --gen_name add_weather_emb_mask_consecutive --cuda 2 --dataset_seed $seed &
    wait
done
