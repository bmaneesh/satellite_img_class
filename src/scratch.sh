#!/bin/bash
#SBATCH -A research
#SBATCH -c 6
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=2048
#SBATCH --time=15:00:00
#SBATCH --mail-type=END
#SBATCH --workdir="/home/maneesh/satellite/src"

module add cuda/8.0
module add cudnn/7-cuda-8.0


python main_all.py
#srun -c 6 --mem-per-cpu=2048 --time=10:00:00 --gres=gpu:1 python main_all.py
#tensorboard --logdir = /home/maneesh/$(ls -t ../logs | head -1) ssh -N -f -R 6080:localhost:6006 maneesh@10.1.66.194
#wait = $(sleep 60)
#id1 = $(srun --dependency = afterok:$wait tensorboard --logdir = /home/maneesh/$(ls -t ../logs | head -1))
#wait1 = $(sleep 60)
#id2 = $(srun --dependecy = afterok:$wait1 ssh -N -f -R 6080:localhost:6006 maneesh@10.1.66.194)