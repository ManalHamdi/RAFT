#SBATCH -J "RAFT"   # job name
#SBATCH --mail-user=manal.hamdi@tum.de   # email address
#SBATCH --mail-type=END  # send at the end
#SBATCH --ntasks=4   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --gres=gpu:1  # gpus if needed
# run your program here

module load python/anaconda3
conda activate raft
mkdir -p checkpoints
CUDA_VISIBLE_DEVICES=4
/home/guests/manal_hamdi/.conda/envs/raft/bin/python -u train_new.py --name raft-acdc --stage acdc --validation acdc --dataset_folder "/home/guests/manal_hamdi/manal/RAFT/datasets/ACDC_processed/" --num_steps 100 --gpus 0 --batch_size 1 --lr 0.0004 --wdecay 0.0001 --max_seq_len 20 --beta_spatial 0.0 --beta_temporal 10.0 #--restore_ckpt "checkpoints/1_raft-acdc.pth"
 
