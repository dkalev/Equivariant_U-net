#!/bin/bash
#SBATCH --job-name="DRIVE_EquivUNet"
#SBATCH --nodes=1 # Number of nodes
#SBATCH --cpus-per-task=3
#SBATCH --ntasks-per-node=1
#SBATCH --time=04:00:00 # excepted wall clock time
#SBATCH --partition=gpu_shared # specify partition

module load 2020
module load Anaconda3/2020.02
module load CUDA/11.0.2-GCC-9.3.0

cp -r $HOME/DRIVE "$TMPDIR" # copy data to scratch

cd "$TMPDIR"/DRIVE

python train.py

cp -r lightning_logs/* $HOME/DRIVE/lightning_logs
echo DONE