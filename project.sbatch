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

nvcc cuda_bellman_ford.cu -o cuda_bellman_ford
gcc -fopenmp bellman_ford_omp.c -o bellman_ford_omp
gcc compare_omp_cuda.c -o compare_omp_cuda

./cuda_bellman_ford london_temporal_at_23.csv 256 1
./bellman_ford_omp
./compare_omp_cuda