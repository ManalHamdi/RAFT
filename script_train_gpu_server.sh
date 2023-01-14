#!/bin/sh
#SBATCH -J "setting2"   # job name
#SBATCH --mail-user=manal.hamdi@tum.de   # email address
#SBATCH --mail-type=END  # send at the end
#SBATCH --ntasks=4   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --gres=gpu:1  # gpus if needed
#SBATCH --output=/home/guests/manal_hamdi/manal/TGRAFT/output_files/output3.out
#SBATCH --error=/home/guests/manal_hamdi/manal/TGRAFT/output_files/error3.rr  
##SBATCH --nodelist=c1-head

# run your program here

module load python/anaconda3
conda activate raftnew
conda info --envs

#python3 -u train.py --experiment "GroupwiseFull_100spa" 
#python3 -u train.py --experiment "GroupwiseFull_50spa" 
#python3 -u train.py --experiment "GroupwiseFull_20spa" 
#python3 -u train.py --experiment "GroupwiseFull_LearnTemp_100spa_mean_default"
#python -u train.py --experiment "GroupwiseFull_LearnTemp_default_17"
python -u train.py --experiment "GroupwiseFull_LearnTemp_setting3_15"
 
#python -u train.py --experiment "GroupwiseFull_LearnTemp_100spa"