#! /bin/bash
#SBATCH --partition=csedu
#SBATCH --time=0-00:05:00
echo "activate env"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate scriptie3
cd ~/master/ScriptieRepo/
echo "start script construct_feature_file.py"
python keep_one_frontal_patient.py
