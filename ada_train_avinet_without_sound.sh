#! /bin/bash
#SBATCH -A rafaelgetto
#SBATCH -n 10
#SBATCH --partition=long
#SBATCH --mem-per-cpu=2G
#SBATCH --gres=gpu:2
#SBATCH --nodelist=gnode041
#SBATCH --time=60:00:00

# conda initialization 
source /home2/rafaelgetto/miniconda3/etc/profile.d/conda.sh; 

# activate conda environment 
conda activate smai_proj;
echo "conda environment activated";

echo "Starting Training for AVAD without sound";
python train.py \
--data_directory /ssd_scratch/cvit/rafaelgetto/ \
--dataset AVAD \
--split 1 \
--load_weight ViNet_Logs/ViNet_Logs_ViNet_early_stopping/version_0/checkpoints/epoch\=64-step\=9749-val_Loss\=1.2245-val_cc_loss\=0.5153-val_similarity\=0.4076.ckpt \
--batch_size 2 \
--experiment_name ViNet_without_sound_AVAD_earlystopping;

echo "deactivate environment";
conda deactivate; 
