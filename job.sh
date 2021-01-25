#!/bin/bash
#SBATCH -N 2 # Number of nodes
#SBATCH -t 04:00:00 # excepted wall clock time
#SBATCH -p gpu_shared # specify partition

module load 2020
module load Anaconda3/2020.02
module load CUDA/11.0.2-GCC-9.3.0

cp -r $HOME/DRIVE/training "$TMPDIR" # copy data to scratch

python $HOME/DRIVE/train.py --output_dir="$TMPDIR"

cp -r "$TMPDIR"/models $HOME/DRIVE

echo DONE