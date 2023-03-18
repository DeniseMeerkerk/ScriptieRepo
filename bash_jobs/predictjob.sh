#! /bin/bash
#SBATCH --partition=csedu
#SBATCH --time=00:15:00
echo "activate env"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate scriptie3
cd ~/master/ScriptieRepo/keras-yolo3/
python predict.py -c config.json -i ~/master/ScriptieRepo/object_relation_transformer/vis/imgs/ -o /ceph/csedu-scratch/project/dmeerkerk/UI_Xray/test_yolo_out/
