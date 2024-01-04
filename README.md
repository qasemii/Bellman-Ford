# Bellman-Ford
A parallel Implementation of 	Bellman-Ford using OpenMP and CUDA.

To run the program use the following:

```
sbatch project.sbatch
```

or directly run the following commands:

```
gcc -fopenmp bellman_ford_omp.c -o bellman_ford_omp
nvcc bellman_ford_cuda.cu -o bellman_ford_cuda
gcc compare_omp_cuda.c -o compare_omp_cuda

./bellman_ford_omp
./bellman_ford_cuda
./compare_omp_cuda
```

After successful execution, `omp_output.txt` and `cuda_output.txt` are generated and include the results for each program. Also, the `report.out` gives a summary of the input specifications (number of nodes and edges), OpenMP and CUDA configuration (number of threads, blocks, and threads) and runtimes, and verification of the results by comparing the OpenMP and CUDA outputs.

For more detailed instructions please refer to [project.pdf](https://github.com/qasemii/Bellman-Ford/blob/main/project.pdf).
