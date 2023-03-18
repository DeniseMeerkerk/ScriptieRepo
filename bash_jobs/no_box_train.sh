#!/bin/bash
#SBATCH --partition=csedu
#SBATCH --time=06-16:05:00
#SBATCH --gres=gpu:1
#SBATCH --mail-type=NONE
#SBATCH --mail-user=dmeerkerk@science.ru.nl
# your job code goes here:
nvidia-smi -L
echo "activate env"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate scriptie4
nvidia-smi
cd ~/master/ScriptieRepo/object_relation_transformer/
export CUDA_VISIBLE_DEVICES=0
echo "starting the script"
python train.py --id first_training_all_data \
--caption_model relation_transformer \
--input_json /ceph/csedu-scratch/project/dmeerkerk/UI_Xray/json/train_hopefully_final_out_all_patched_met_elena.json \
--input_fc_dir /ceph/csedu-scratch/project/dmeerkerk/UI_Xray/bu_out_all_no_box_fc \
--input_att_dir /ceph/csedu-scratch/project/dmeerkerk/UI_Xray/bu_out_all_no_box_att \
--input_label_h5 /ceph/csedu-scratch/project/dmeerkerk/UI_Xray/train_out_hopefully_final_label.h5  \
--input_box_dir /ceph/csedu-scratch/project/dmeerkerk/UI_Xray/bu_out_all_no_box_box \
--input_rel_box_dir /ceph/csedu-scratch/project/dmeerkerk/UI_Xray/bu_out_all_no_box_rel \
--checkpoint_path /ceph/csedu-scratch/project/dmeerkerk/UI_Xray/model/no_box_14_dec/ \
--label_smoothing 0.0 \
--batch_size 1 \
--learning_rate 5e-4 \
--num_layers 6 \
--input_encoding_size 512 \
--rnn_size 1024 \
--learning_rate_decay_start 0 \
--scheduled_sampling_start 0 \
--save_checkpoint_every 800 \
--language_eval 0 \
--val_images_use 360 \
--self_critical_after 50 \
--max_epochs 100 \
--use_box 1  \
--seq_per_img 1 \
--cached_tokens /ceph/csedu-scratch/project/dmeerkerk/UI_Xray/train_out116-idxs
--start_from /ceph/csedu-scratch/project/dmeerkerk/UI_Xray/model/no_box_13_dec/

# --input json ...../train_likecoco_out_all_patched.json
# train_likecoco_out.json
