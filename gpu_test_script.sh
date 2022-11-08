#!/usr/bin/bash
#SBATCH -J "eval298"   # job name
#SBATCH --mail-user=manal.hamdi@tum.de   # email address
#SBATCH --mail-type=END  # send at the end
#SBATCH --ntasks=4   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --gres=gpu:1  # gpus if needed
#SBATCH --output=/home/guests/manal_hamdi/manal/TGRAFT/output_files/error_Avg_19_noNorm_full_198.err
#SBATCH --output=/home/guests/manal_hamdi/manal/TGRAFT/output_files/output_Avg_19_noNorm_full_198.out 
# run your program here

module load python/anaconda3
conda deactivate
conda activate raft
echo "Activated raft"

python3 evaluate.py --experiment 'Eval_config_19_GroupwiseFull_198' 