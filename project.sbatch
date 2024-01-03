#!/bin/bash
#SBATCH --job-name=B
#SBATCH --output=report.out
#SBATCH --error=error.err

#SBATCH --mail-type=ALL
#SBATCH --mail-user=mohammadreza.ghasemi@unibo.it
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1

gcc -fopenmp bellman_ford_omp.c -o bellman_ford_omp
nvcc bellman_ford_cuda.cu -o bellman_ford_cuda
gcc compare_omp_cuda.c -o compare_omp_cuda

./bellman_ford_omp 2 #NUM_THREADS
./bellman_ford_cuda 128 #blocksPerGrid, threadsPerBlock
./compare_omp_cuda