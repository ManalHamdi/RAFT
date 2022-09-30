#!/usr/bin/bash
#SBATCH -J "RAFT"   # job name
#SBATCH --mail-user=manal.hamdi@tum.de   # email address
#SBATCH --mail-type=END  # send at the end
#SBATCH --ntasks=4   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --gres=gpu:1  # gpus if needed
#SBATCH --output=/home/guests/manal_hamdi/manal/RAFT/error.err
#SBATCH --output=/home/guests/manal_hamdi/manal/RAFT/output.out 
# run your program here

module load python/anaconda3
conda deactivate
conda activate raft
echo "Activated raft"

mkdir -p checkpoints
python3 -u evaluate.py --dataset 'acdc' --dataset_folder "/home/guests/manal_hamdi/manal/RAFT/datasets/ACDC_processed/"  --gpus 0 --batch_size 1  --max_seq_len 8 --gamma 0.8 --restore_ckpt "new_checkpoints/200_noNorm_19_all"