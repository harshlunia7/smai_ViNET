#! /bin/bash
#SBATCH -A rafaelgetto
#SBATCH -n 10
#SBATCH --partition=long
#SBATCH --mem-per-cpu=2G
#SBATCH --gres=gpu:2
#SBATCH --nodelist=gnode026
#SBATCH --time=60:00:00

# conda initialization 
source /home2/rafaelgetto/miniconda3/etc/profile.d/conda.sh; 

# activate conda environment 
conda activate smai_proj;
echo "conda environment activated";

# echo "Creating ssd_scratch/cvit/rafaelgetto directory";
# mkdir /ssd_scratch/cvit/rafaelgetto;
# mkdir /ssd_scratch/cvit/rafaelgetto/smai_proj_vinet;
# 
# copy dataset to ssd_scratch 
# copying tar.gz file to ssd_scratch from share3/rafaelgetto
# rsync -aP rafaelgetto@ada.iiit.ac.in:/share3/rafaelgetto/small_dhf1k.tar.gz /ssd_scratch/cvit/rafaelgetto/smai_proj_vinet;
# echo "Dataset Copied";
# tar --warning=none -xzf  /ssd_scratch/cvit/rafaelgetto/smai_proj_vinet/small_dhf1k.tar.gz -C /ssd_scratch/cvit/rafaelgetto/smai_proj_vinet;
# echo "untar finished";

# rm /ssd_scratch/cvit/rafaelgetto/smai_proj_vinet/small_dhf1k.tar.gz;
# echo "Removed tar file from ssd_scratch"

echo "Starting Training";
python train.py \
--data_directory /ssd_scratch/cvit/rafaelgetto/ \
--dataset DIEM \
--split -1 \
--use_sound \
--use_transformer \
--batch_size 2 \
--load_weight ViNet_Logs/ViNet_Logs_ViNet_epoch_100/version_0/checkpoints/epoch\=105-step\=15899.ckpt \
--fusing_method bilinear \
--experiment_name AViNet_epoch_105_bilinear;

python train.py \
--data_directory /ssd_scratch/cvit/rafaelgetto/ \
--dataset DIEM \
--split -1 \
--use_sound \
--use_transformer \
--batch_size 2 \
--load_weight ViNet_Logs/ViNet_Logs_ViNet_epoch_100/version_0/checkpoints/epoch\=105-step\=15899.ckpt \
--fusing_method concat \
--experiment_name AViNet_epoch_105_concat;

# echo "Removing the dataset from ssd_scratch"
# rm -rf /ssd_scratch/cvit/rafaelgetto/parseq_gujarati;

#echo "copying remaining files from ssd_scratch to home2/rafaelgetto/results"
#rsync -azP /ssd_scratch/cvit/rafaelgetto/ rafaelgetto@ada.iiit.ac.in:/home2/rafaelgetto/results/
#echo "Copied Files to home2/rafaelgetto/results"

echo "deactivate environment";
conda deactivate; 
