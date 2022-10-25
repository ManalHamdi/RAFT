#!/bin/sh
#SBATCH -J "diff_sched"   # job name
#SBATCH --mail-user=manal.hamdi@tum.de   # email address
#SBATCH --mail-type=END  # send at the end
#SBATCH --ntasks=4   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --gres=gpu:1  # gpus if needed
#SBATCH --nodelist=c1-head

# run your program here

conda activate raft

python3 -u train.py --experiment "PairwiseFull"
 
