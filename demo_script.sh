#!/bin/bash
#SBATCH -J “RAFT_demo"   # job name
#SBATCH --mail-user=manal.hamdi@tum.de   # email address
#SBATCH --mail-type=END  # send at the end
#SBATCH --ntasks=3   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --gres=gpu:1  # gpus if needed

# run your program here

module load python/anaconda3
conda activate raft
python3 demo.py --model=models/raft-things.pth --path=demo-frames
