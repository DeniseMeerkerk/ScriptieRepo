#! /bin/bash
#SBATCH --partition=csedu
#SBATCH --time=0-00:35:00
echo "activate env"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate scriptie3
# rm -r /ceph/csedu-scratch/project/dmeerkerk/UI_Xray/bu_ou*
cd ~/master/ScriptieRepo/object_relation_transformer/scripts
echo "start script"
python make_bu_data.py \
--downloaded_feats /ceph/csedu-scratch/project/dmeerkerk/UI_Xray/feature_files/thresh/ \
--output_dir /ceph/csedu-scratch/project/dmeerkerk/UI_Xray/bu_out_all_thresh \
--tsv_file features_combined_0-3689.tsv
