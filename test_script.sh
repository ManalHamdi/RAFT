#!/bin/sh
#SBATCH -J "eval"   # job name
#SBATCH --mail-user=manal.hamdi@tum.de   # email address
#SBATCH --mail-type=END  # send at the end
#SBATCH --ntasks=4   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --gres=gpu:1  # gpus if needed

# run your program here

conda activate raft

#python3 evaluate.py --experiment "Eval25_19config_GroupwiseFull_200"
#python3 evaluate.py --experiment "Eval25_19config_GroupwisePhoto_200"
#python3 evaluate.py --experiment "Eval25_19config_GroupwisePhotoSpa_200"
#python3 evaluate.py --experiment "Eval25_19config_GroupwisePhotoTemp_200"
#python3 evaluate.py --experiment "Eval25_19config_PairwiseFwd400"
#python3 evaluate.py --experiment "Eval25_19config_GroupwiseFull_358"
#python3 evaluate.py --experiment "Eval25_19config_PairwiseFwd358"
#python3 evaluate.py --experiment "Eval8_19config_GroupwiseFull_358"
#python3 evaluate.py --experiment "Eval8_19config_PairwiseFull_358"
#python3 evaluate.py --experiment "Eval19_19config_GroupwiseFull_358"
#python3 evaluate.py --experiment "Eval19_19config_PairwiseFull_358"

#python3 evaluate.py --experiment "Eval25_19config_PairwiseFwd400"
#python3 evaluate.py --experiment "Eval25_19config_GroupwiseFull_398"
#python3 evaluate.py --experiment "Eval8_19config_GroupwiseFull_398"
#python3 evaluate.py --experiment "Eval8_19config_PairwiseFull_398"

#python3 evaluate.py --experiment "Eval19_19config_GroupwiseFull_398"
python3 evaluate.py --experiment "Eval25_19config_GroupwiseFullComp_68"

#python3 evaluate.py --experiment "Eval8_4config_GroupwiseFull_398"
#python3 evaluate.py --experiment "Eval8_4config_PairwiseFull_398"
#python3 evaluate.py --experiment "Eval19_4config_GroupwiseFull_398"
#python3 evaluate.py --experiment "Eval19_4config_PairwiseFull_369"
