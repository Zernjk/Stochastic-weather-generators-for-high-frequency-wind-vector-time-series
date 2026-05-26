#!/bin/bash

gen_names=(generated_unconditional generated_consecutive add_weather_unconditional add_weather_consecutive add_weather_emb_mask_unconditional add_weather_emb_mask_consecutive)

for gen_name in "${gen_names[@]}"; do
    for seed in $(seq 42 10 91); do
        echo "Running with gen_name: $gen_name, seeds: $seed, $((seed+1)), $((seed+2)), $((seed+3)), $((seed+4)), $((seed+5)), $((seed+6)), $((seed+7)), $((seed+8)), $((seed+9))"
        ~/.conda/envs/deeplearning/bin/python source.py --real_name training --gen_name $gen_name --cuda 3 --dataset_seed $seed &
        ~/.conda/envs/deeplearning/bin/python source.py --real_name training --gen_name $gen_name --cuda 3 --dataset_seed $((seed+1)) &
        ~/.conda/envs/deeplearning/bin/python source.py --real_name training --gen_name $gen_name --cuda 3 --dataset_seed $((seed+2)) &
        ~/.conda/envs/deeplearning/bin/python source.py --real_name training --gen_name $gen_name --cuda 3 --dataset_seed $((seed+3)) &
        ~/.conda/envs/deeplearning/bin/python source.py --real_name training --gen_name $gen_name --cuda 3 --dataset_seed $((seed+4)) &
        ~/.conda/envs/deeplearning/bin/python source.py --real_name training --gen_name $gen_name --cuda 3 --dataset_seed $((seed+5)) &
        ~/.conda/envs/deeplearning/bin/python source.py --real_name training --gen_name $gen_name --cuda 3 --dataset_seed $((seed+6)) &
        ~/.conda/envs/deeplearning/bin/python source.py --real_name training --gen_name $gen_name --cuda 3 --dataset_seed $((seed+7)) &
        ~/.conda/envs/deeplearning/bin/python source.py --real_name training --gen_name $gen_name --cuda 3 --dataset_seed $((seed+8)) &
        ~/.conda/envs/deeplearning/bin/python source.py --real_name training --gen_name $gen_name --cuda 3 --dataset_seed $((seed+9)) &
        wait
    done
done
