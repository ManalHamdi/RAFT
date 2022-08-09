#SBATCH -J "RAFT"   # job name
#SBATCH --mail-user=manal.hamdi@tum.de   # email address
#SBATCH --mail-type=END  # send at the end
#SBATCH --ntasks=4   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --gres=gpu:1  # gpus if needed
#SBATCH --nodelist=c1-node01
# run your program here

module load python/anaconda3
conda activate raft
mkdir -p checkpoints
CUDA_VISIBLE_DEVICES=7
python3 -u train_new.py --name raft-acdc --stage acdc --validation acdc --dataset_folder "/home/kevin/manal/RAFT/datasets/ACDC_processed/" --gpus 0 --num_steps 50 --batch_size 1 --lr 0.0004 --image_size 368 496 --wdecay 0.0001
#python3 -u train.py --name raft-chairs --stage chairs --validation chairs --gpus 0 1 --num_steps 100000 --batch_size 10 --lr 0.0004 --image_size 368 496 --wdecay 0.0001
