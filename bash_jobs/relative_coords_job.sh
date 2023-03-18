#! /bin/bash
#SBATCH --partition=csedu
#SBATCH --time=0-00:25:00
echo "activate env"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate scriptie3
cd ~/master/ScriptieRepo/object_relation_transformer/scripts
echo "start script"
python prepro_bbox_relative_coords.py \
--input_json /ceph/csedu-scratch/project/dmeerkerk/UI_Xray/json/train_hopefully_final.json  \
--input_box_dir /ceph/csedu-scratch/project/dmeerkerk/UI_Xray/bu_out_all_thresh_box/ \
--output_dir /ceph/csedu-scratch/project/dmeerkerk/UI_Xray/bu_out_all_thresh_rel/ \
--image_root /ceph/csedu-scratch/project/dmeerkerk/UI_Xray/images/images_normalized/


