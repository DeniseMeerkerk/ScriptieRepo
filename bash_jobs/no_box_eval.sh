#!/bin/bash
#SBATCH --partition=csedu
#SBATCH --time=00:05:00
#SBATCH --gres=gpu:1
#SBATCH --mail-type=NONE
#SBATCH --mail-user=dmeerkerk@science.ru.nl
# your job code goes here:
nvidia-smi -L
echo "activate env"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate scriptie3
cd ~/master/ScriptieRepo/object_relation_transformer/
export CUDA_VISIBLE_DEVICES=0
echo "starting the script"
python eval.py \
        --dump_images 0 \
        --num_images -1 \
        --model /ceph/csedu-scratch/project/dmeerkerk/UI_Xray/model/no_box_14_dec/model-best.pth \
        --infos_path /ceph/csedu-scratch/project/dmeerkerk/UI_Xray/model/no_box_14_dec/infos_first_training_all_data-best.pkl \
        --image_root /ceph/csedu-scratch/project/dmeerkerk/UI_Xray/images/images_normalized/ \
        --input_json /ceph/csedu-scratch/project/dmeerkerk/UI_Xray/json/train_hopefully_final_out_all_patched_met_elena.json \
        --input_fc_dir /ceph/csedu-scratch/project/dmeerkerk/UI_Xray/bu_out_all_no_box_fc \
        --input_att_dir /ceph/csedu-scratch/project/dmeerkerk/UI_Xray/bu_out_all_no_box_att \
        --input_box_dir /ceph/csedu-scratch/project/dmeerkerk/UI_Xray/bu_out_all_no_box_box \
        --input_rel_box_dir=/ceph/csedu-scratch/project/dmeerkerk/UI_Xray/bu_out_all_no_box_rel \
        --input_label_h5 /ceph/csedu-scratch/project/dmeerkerk/UI_Xray/train_out_hopefully_final_label.h5  \
        --language_eval 0 \
	--experiment no_box_14_dec
