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
CUDA_VISIBLE_DEVICES=1
python3 -u train_new.py --name raft-acdc --stage acdc --validation acdc --dataset_folder "/home/kevin/manal/RAFT/datasets/ACDC_processed/" --num_steps 100 --gpus 1 --batch_size 1 --lr 0.0004 --image_size 256 256 --wdecay 0.0001 #--restore_ckpt "runs/Aug08_13-12-09_bagger/events.out.tfevents.1659957129.bagger.861966.0"

#python3 -u train.py --name raft-chairs --stage chairs --validation chairs --gpus 0 1 --num_steps 100000 --batch_size 10 --lr 0.0004 --image_size 368 496 --wdecay 0.0001
