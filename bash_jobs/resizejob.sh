#! /bin/bash
#SBATCH --partition=csedu
#SBATCH --time=8:00:00
echo "activate env"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate scriptie3
python ~/master/ScriptieRepo/resize_dicom_images.py
