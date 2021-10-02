#!/bin/bash
#SBATCH -p gpu
#SBATCH -c 4 
#SBATCH --gres=gpu:1 


if [ "$#" -ne 3 ]; then
    echo "Please, provide the the training framework: torch/tf, dataset path and device"
    exit 1
fi

cd ../..
python scripts/run_pipeline.py $1 -c ml3d/configs/randlanet_electricity3d.yml \
--dataset_path $2 --device $3
