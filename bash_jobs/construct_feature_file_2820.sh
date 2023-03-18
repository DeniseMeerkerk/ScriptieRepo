#! /bin/bash
#SBATCH --partition=csedu
#SBATCH --time=01-00:10:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dmeerkerk@science.ru.nl
echo "activate env"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate scriptie3
cd ~/master/ScriptieRepo/keras-yolo3/
echo "start script construct_feature_file.py"
python construct_feature_file.py \
-w /ceph/csedu-scratch/project/dmeerkerk/XRAY_vinbig_png15000.h5 \
-i /ceph/csedu-scratch/project/dmeerkerk/UI_Xray/images/images_normalized/ \
-t /ceph/csedu-scratch/project/dmeerkerk/UI_Xray/feature_files/constructed_feats_no_box_test.tsv \
-s 3480 \
--slice 40 \
--no_box True
# -i /ceph/csedu-scratch/project/dmeerkerk/UI_Xray/subset30/
# 
