#!/bin/bash
#SBATCH --partition=csedu
#SBATCH --time=01:05:00
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
        --model /ceph/csedu-scratch/project/dmeerkerk/UI_Xray/model/exp_up10_02_dec/model-best.pth \
        --infos_path /ceph/csedu-scratch/project/dmeerkerk/UI_Xray/model/exp_up10_02_dec/infos_first_training_all_data-best.pkl \
        --image_root /ceph/csedu-scratch/project/dmeerkerk/UI_Xray/images/images_normalized/ \
        --input_json /ceph/csedu-scratch/project/dmeerkerk/UI_Xray/json/train_likecoco_out_all_patched_met_elena_up10.json \
        --input_fc_dir /ceph/csedu-scratch/project/dmeerkerk/UI_Xray/bu_out_all_fc2 \
        --input_att_dir /ceph/csedu-scratch/project/dmeerkerk/UI_Xray/bu_out_all_att \
        --input_box_dir /ceph/csedu-scratch/project/dmeerkerk/UI_Xray/bu_out_all_box \
        --input_rel_box_dir=/ceph/csedu-scratch/project/dmeerkerk/UI_Xray/bu_out_all_rel \
        --input_label_h5 /ceph/csedu-scratch/project/dmeerkerk/UI_Xray/train_out_up10_label.h5  \
        --language_eval 0 \
	--experiment exp_up_10_02_dec
