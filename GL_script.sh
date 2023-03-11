#!/bin/bash
#SBATCH --account=eecs448w23_class
#SBATCH --partition=standard
#SBATCH --time=00-02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16 
#SBATCH --mem-per-gpu=32GB
# set up job
module load python/3.9.12
source ~/env/bin/activate

which python3

# run job
python gridsearch.py
