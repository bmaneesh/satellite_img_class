#!/bin/bash
#SBATCH -A research
#SBATCH -c 6
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=2048
#SBATCH --time=15:00:00
#SBATCH --mail-type=END
#SBATCH --workdir="./"

module add cuda/8.0
module add cudnn/7-cuda-8.0


python main_all.py
