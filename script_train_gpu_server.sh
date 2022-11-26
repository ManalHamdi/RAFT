#!/bin/sh
#SBATCH -J "compall100"   # job name
#SBATCH --mail-user=manal.hamdi@tum.de   # email address
#SBATCH --mail-type=END  # send at the end
#SBATCH --ntasks=4   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --gres=gpu:1  # gpus if needed
#SBATCH --output=/home/guests/manal_hamdi/manal/TGRAFT/output_files/output2.out
#SBATCH --error=/home/guests/manal_hamdi/manal/TGRAFT/output_files/error2.err  
##SBATCH --nodelist=c1-node01

# run your program here

module load python/anaconda3
conda activate raftnew
conda info --envs

#python3 -u train.py --experiment "GroupwiseFull_100spa" 
#python3 -u train.py --experiment "GroupwiseFull_50spa" 
#python3 -u train.py --experiment "GroupwiseFull_20spa" 
#python3 -u train.py --experiment "GroupwiseFull_compFlow"
#python -u train.py --experiment "GroupwiseFull_compFlow_100spa"
 
python -u train.py --experiment "GroupwiseFull_LearnFlow_100spa"