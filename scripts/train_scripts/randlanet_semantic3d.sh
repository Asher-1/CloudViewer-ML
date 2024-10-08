#!/bin/bash
#SBATCH -p gpu
#SBATCH -c 4 
#SBATCH --gres=gpu:1 


if [ "$#" -ne 4 ]; then
    echo "Please, provide the the training framework: torch/tf and dataset path"
    exit 1
fi

cd ../..
python scripts/run_pipeline.py $1 -c ml3d/configs/randlanet_semantic3d.yml \
--dataset_path $2 --ckpt_path $3 --split $4
