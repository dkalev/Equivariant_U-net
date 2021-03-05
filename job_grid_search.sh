#!/bin/bash
#SBATCH --job-name="Search"
#SBATCH --nodes=1 # Number of nodes
#SBATCH --time=04:00:00 # excepted wall clock time
#SBATCH --partition=gpu_shared # specify partition
#SBATCH --signal=SIGUSR1@90 #enables pl to save a checkpoint if the job is to be terminated
#SBATCH --output=out/%x.%j.out

module load 2020
module load Anaconda3/2020.02
module load CUDA/11.0.2-GCC-9.3.0

export SLURM_JOB_NAME=bash # hack to make pl + ray tune work on slurm

source activate dl

cp -r $HOME/DRIVE "$TMPDIR" # copy data to scratch

cd "$TMPDIR"/DRIVE

echo "Start training model $1"
srun python -u grid_search.py --model=$1

cp -r ray_tune/* $HOME/DRIVE/ray_tune
echo DONE