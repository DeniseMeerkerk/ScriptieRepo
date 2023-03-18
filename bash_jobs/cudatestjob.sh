#!/bin/bash
#SBATCH --partition=csedu
#SBATCH --gres=gpu:1
#SBATCH --time=00:02:00
# your job code goes here:
nvidia-smi -L
echo "activate env"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate scriptie3
cd ~/master
echo "starting the script"
python cudatest.py

