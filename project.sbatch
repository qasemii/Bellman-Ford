#!/bin/bash
#SBATCH --job-name=buda
#SBATCH --output=buda.out
#SBATCH --error=buda.err

#SBATCH --mail-type=ALL
#SBATCH --mail-user=mohammadreza.ghasemi@unibo.it
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1

nvcc -o buda buda.cu
./buda