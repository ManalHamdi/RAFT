#!/usr/bin/bash
#SBATCH -J "RAFT"   # job name
#SBATCH --mail-user=manal.hamdi@tum.de   # email address
#SBATCH --mail-type=END  # send at the end
#SBATCH --ntasks=4   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --gres=gpu:1  # gpus if needed
# run your program here

module load python/anaconda3
conda deactivate
conda activate raftkevin
echo "Activated raftkevin"
mkdir -p checkpoints
CUDA_VISIBLE_DEVICES=0
python3 -u train.py --name raft-acdc --stage acdc --validation acdc --dataset_folder "/home/kevin/manal/RAFT/datasets/ACDC_processed/" --num_steps 300 --gpus 1 --batch_size 1 --lr 0.0004 --wdecay 0.0001 --max_seq_len 20 --gamma 0.8 --beta_photo 1.0 --beta_spatial 10.0 --beta_temporal 10.0 #--restore_ckpt "checkpoints/1_raft-acdc.pth"