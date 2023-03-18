#! /bin/bash
#SBATCH --partition=csedu
#SBATCH --time=00:05:00
echo "activate env"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate scriptie3
cd ~/master/ScriptieRepo/
python object_relation_transformer/scripts/prepro_labels.py \
--input_json /ceph/csedu-scratch/project/dmeerkerk/UI_Xray/json/train_hopefully_finalup10.json \
--output_json /ceph/csedu-scratch/project/dmeerkerk/UI_Xray/json/train_dont_use.json \
--output_h5 /ceph/csedu-scratch/project/dmeerkerk/UI_Xray/train_out_up10 \
--word_count_threshold 3 \
--max_length 116
