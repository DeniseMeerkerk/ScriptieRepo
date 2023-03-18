#! /bin/bash
#SBATCH --partition=csedu
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
nvidia-smi -L
echo "remove env scriptie3 to make room for env scriptie2"
source ~/miniconda3/etc/profile.d/conda.sh
# conda activate scriptie3
cd ~/master/
# conda env export > scriptie3.yml
# conda deactivate
rm -r ~/miniconda2/envs/scriptie3/

wait

echo "build scriptie2 env"
cd ~/master/
conda env create -f scriptie2.yml

wait

echo "activate scriptie2 env"
conda activate scriptie2

echo "run script"
cd ~/master/ScriptieRepo/object_relation_transformer/scripts/
python prepro_feats.py \
--input_json /ceph/csedu-scratch/project/dmeerkerk/UI_Xray/json/train_likecoco.json \
--output_dir /ceph/csedu-scratch/project/dmeerkerk/UI_Xray/prepro_output \
--model_root /ceph/csedu-scratch/project/dmeerkerk/UI_Xray/imagenet_weights

wait

echo "remove env scriptie2 and reinstall env scriptie3"
conda deactivate
rm -r ~/miniconda2/envs/scriptie2/
wait
cd ~/master/
conda env create -f scriptie3.yml
wait
