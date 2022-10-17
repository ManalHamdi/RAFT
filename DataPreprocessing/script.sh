#!/bin/sh
#SBATCH -J "diff_sched"   # job name
#SBATCH --mail-user=manal.hamdi@tum.de   # email address
#SBATCH --mail-type=END  # send at the end
#SBATCH --ntasks=4   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --gres=gpu:1  # gpus if needed
#SBATCH --output=/home/guests/manal_hamdi/manal/RAFT/output_files/output.out
#SBATCH --error=/home/guests/manal_hamdi/manal/RAFT/output_files/error.err  
##SBATCH --nodelist=c1-head

# run your program here

module load python/anaconda3
conda activate generate

/home/guests/manal_hamdi/.conda/envs/raft/bin/python -u DataPreprocessing/ACDC_Preprocess_Script.py --acdc_folder "datasets/ACDC/" --pair_folder "datasets/ACDC_pairs/validation/"