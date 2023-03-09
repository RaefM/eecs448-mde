#!/bin/bash
#SBATCH --account=eecs448w23_class
#SBATCH --partition=spgpu
#SBATCH --time=00-00:30:00
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=32GB
#SBATCH --get-user-env
# set up job
module load python/3.9.12 cuda
source /home/kshenton/
pushd /home/kshenton

which python3

# run job
python3 gridsearch.py
