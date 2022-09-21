#!/bin/sh
#SBATCH -J "RAFT"   # job name
#SBATCH --mail-user=manal.hamdi@tum.de   # email address
#SBATCH --mail-type=END  # send at the end
#SBATCH --ntasks=4   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --gres=gpu:1  # gpus if needed
#SBATCH --output=/home/guests/manal_hamdi/manal/RAFT/output_files/output_lr.out  # Standard output of the script (Can be absolute or relative path)
#SBATCH --error=/home/guests/manal_hamdi/manal/RAFT/output_files/error_lr.err  # Standard error of the script
# run your program here

module load python/anaconda3
conda activate raft
mkdir -p checkpoints
/home/guests/manal_hamdi/.conda/envs/raft/bin/python -u train.py --name raft-acdc --stage acdc --validation acdc --dataset_folder "/home/guests/manal_hamdi/manal/RAFT/datasets/ACDC_processed/" --num_steps 200 --gpus 0 --batch_size 1 --lr 0.000004 --wdecay 0.0001 --max_seq_len 19 --add_normalisation --beta_photo 1 --beta_spatial 10.0 --beta_temporal 10.0 #--restore_ckpt "checkpoints/1_raft-acdc.pth" --add_normalisation
 
