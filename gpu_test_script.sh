#!/usr/bin/bash
#SBATCH -J "eval"   # job name
#SBATCH --mail-user=manal.hamdi@tum.de   # email address
#SBATCH --mail-type=END  # send at the end
#SBATCH --ntasks=4   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --gres=gpu:1  # gpus if needed
#SBATCH --output=/home/guests/manal_hamdi/manal/RAFT/output_files/error_Avg_19_noNorm_full_198.err
#SBATCH --output=/home/guests/manal_hamdi/manal/RAFT/output_files/output_Avg_19_noNorm_full_198.out 
# run your program here

module load python/anaconda3
conda deactivate
conda activate raft
echo "Activated raft"

python3 evaluate.py --name 2_Avg_19_noNorm_full_198 --dataset acdc --dataset_folder "/home/guests/manal_hamdi/manal/RAFT/datasets/ACDC_processed/" --gpus 0 --batch_size 1 --max_seq_len 8 --gamma 0.8 --restore_ckpt "october_checkpoints/2_Avg_19_noNorm_full/2_Avg_19_noNorm_full_198.pth" --output_file "debugging_exp_eval_results.txt"