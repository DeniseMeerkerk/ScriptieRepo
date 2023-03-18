#!/bin/bash
#SBATCH --partition=csedu
#SBATCH --gres=gpu:1
#SBATCH --time=02-00:00:00
# your job code goes here:
nvidia-smi -L
echo "activate env"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate scriptie3
cd ~/master/ScriptieRepo/keras-yolo3/
export CUDA_VISIBLE_DEVICES=0
# rm *.pkl
echo "starting the script"
python train.py -c config.json


