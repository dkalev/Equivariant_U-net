#!/bin/bash
#SBATCH --job-name="DRIVE_EquivUNet"
#SBATCH --nodes=1 # Number of nodes
#SBATCH --time=02:00:00 # excepted wall clock time
#SBATCH --partition=gpu_shared # specify partition
#SBATCH --signal=SIGUSR1@90 #enables pl to save a checkpoint if the job is to be terminated

module load 2020
module load Anaconda3/2020.02
module load CUDA/11.0.2-GCC-9.3.0

source activate dl

cp -r $HOME/DRIVE "$TMPDIR" # copy data to scratch

cd "$TMPDIR"/DRIVE

srun python -u train.py

cp -r lightning_logs/* $HOME/DRIVE/lightning_logs
echo DONE