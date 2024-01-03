#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <limits.h>
#include <assert.h>
#include <time.h>
// #include "hpc.h"

#define INF 999999
#define VERTICES 10000 //total vertices 264,346


double gettime(void){
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts );
    return (ts.tv_sec + (double)ts.tv_nsec / 1e9);
}

void read_file(const char* filename, int* weights, int* n_edges) {
    // Initialize the matrix with INF and 0 for diagonals
    for (int i = 0; i < VERTICES; i++) {
        for (int j = 0; j < VERTICES; j++) {
            if (i != j) {
                weights[i * VERTICES + j] = INF;
            } else {
                weights[i * VERTICES + j] = 0;
            }
        }
    }

    // Open the CSV file
    FILE* file = fopen(filename, "r");

    // Read each line in the CSV file and update the matrix
    char line[256];
    n_edges = 0;
    while (fgets(line, sizeof(line), file)) {
        char* token;
        char* rest = line;
        int src_id, dest_id, distance;

        // Tokenize the line based on the comma delimiter
        token = strtok_r(rest, ",", &rest);
        src_id = atoi(token);

        token = strtok_r(rest, ",", &rest);
        dest_id = atoi(token);

        token = strtok_r(rest, ",", &rest);
        distance = atoi(token);

        // Update the matrix with the distance value
        if (src_id < VERTICES && dest_id < VERTICES) {
            n_edges++;
            weights[src_id * VERTICES + dest_id] = distance;
        }
    }
    fclose(file);
}

void save_results(int *distance, bool has_negative_cycle) {
    FILE *outputf = fopen("cuda_output.txt", "w");
    if (!has_negative_cycle) {
        for (int i = 0; i < VERTICES; i++) {
            if (distance[i] > INT_MAX)
                distance[i] = INT_MAX;
            fprintf(outputf, "%d\n", distance[i]);
        }
        fflush(outputf);
    } else {
        fprintf(outputf, "Negative cycle detected!\n");
    }
    fclose(outputf);
}

// sequential ================================================================================================
__global__ void bellman_ford_sequential_kernel(int *d_weights, int *d_distance, bool *d_changed) {
    
    for (int u = 0; u < VERTICES; u++) {
        for (int v = 0; v < VERTICES; v++) {
            int weight = d_weights[u * VERTICES + v];
            if (weight < INF) {
                int new_distance = d_distance[u] + weight;
                if (new_distance < d_distance[v]) {
                    *d_changed = true;
                    d_distance[v] = new_distance;
                }
            }
        }
    }
}

void bellman_ford_sequential(int *weights, int *distance, int start, bool *has_negative_cycle) {

    int iter_num = 0;
    int *d_weights, *d_distance;
    bool *d_changed, h_changed;

    // initializing the distance array
    for (int i = 0; i < VERTICES; i++) {
        distance[i] = INF;
    }
    distance[start] = 0;

    // Allocate GPU memory for d_weights, d_distance, d_changed
    cudaMalloc(&d_weights, sizeof(int) * VERTICES * VERTICES);
    cudaMalloc(&d_distance, sizeof(int) * VERTICES);
    cudaMalloc(&d_changed, sizeof(bool));

    //Transfer the data from host to GPU.
    cudaMemcpy(d_weights, weights, sizeof(int) * VERTICES * VERTICES, cudaMemcpyHostToDevice);
    cudaMemcpy(d_distance, distance, sizeof(int) * VERTICES, cudaMemcpyHostToDevice);

    for (;;) {
        h_changed = false;
        cudaMemcpy(d_changed, &h_changed, sizeof(bool), cudaMemcpyHostToDevice);

        bellman_ford_sequential_kernel<<<1, 1>>>(d_weights, d_distance, d_changed);
        cudaDeviceSynchronize();
        cudaMemcpy(&h_changed, d_changed, sizeof(bool), cudaMemcpyDeviceToHost);

        iter_num++;
        if (iter_num >= VERTICES - 1) {
            *has_negative_cycle = true;
            break;
        }
        if (!h_changed) {
            break;
        }
    }
    if (!*has_negative_cycle) {
        // Copy the shortest path distances back to the host memory
        cudaMemcpy(distance, d_distance, sizeof(int) * VERTICES, cudaMemcpyDeviceToHost);
    }
    
    // Free up the GPU memory.
    cudaFree(d_weights);
    cudaFree(d_distance);
    cudaFree(d_changed);
}

// bellman_ford_withBlocks ===================================================================================
__global__ void bellman_ford_withBlock_kernel(int *d_weights, int *d_distance, bool *d_changed) {
    int global_tid = blockIdx.x;

    if (global_tid < VERTICES){
        for (int u = 0; u < VERTICES; u++) {
            for (int v = global_tid; v < VERTICES; v += gridDim.x) {
                int weight = d_weights[u * VERTICES + v];
                if (weight < INF) {
                    int new_distance = d_distance[u] + weight;
                    if (new_distance < d_distance[v]) {
                        *d_changed = true;
                        d_distance[v] = new_distance;
                    }
                }
            }
        }
    }
}

void bellman_ford_withBlock(int *weights, int *distance, int start, bool *has_negative_cycle) {
    int iter_num = 0;
    int *d_weights, *d_distance;
    bool *d_changed, h_changed;

    // initializing the distance array
    for (int i = 0; i < VERTICES; i++) {
        distance[i] = INF;
    }
    distance[start] = 0;

    // Allocate GPU memory for d_weights, d_distance, d_changed
    cudaMalloc(&d_weights, sizeof(int) * VERTICES * VERTICES);
    cudaMalloc(&d_distance, sizeof(int) * VERTICES);
    cudaMalloc(&d_changed, sizeof(bool));

    //Transfer the data from host to GPU.
    cudaMemcpy(d_weights, weights, sizeof(int) * VERTICES * VERTICES, cudaMemcpyHostToDevice);
    cudaMemcpy(d_distance, distance, sizeof(int) * VERTICES, cudaMemcpyHostToDevice);

    for (;;) {
        h_changed = false;
        cudaMemcpy(d_changed, &h_changed, sizeof(bool), cudaMemcpyHostToDevice);

        bellman_ford_withBlock_kernel<<<VERTICES, 1>>>(d_weights, d_distance, d_changed);
        cudaDeviceSynchronize();
        cudaMemcpy(&h_changed, d_changed, sizeof(bool), cudaMemcpyDeviceToHost);

        iter_num++;
        if (iter_num >= VERTICES - 1) {
            *has_negative_cycle = true;
            break;
        }
        if (!h_changed) {
            break;
        }
    }
    if (!*has_negative_cycle) {
        // Copy the shortest path distances back to the host memory
        cudaMemcpy(distance, d_distance, sizeof(int) * VERTICES, cudaMemcpyDeviceToHost);
    }
    
    // Free up the GPU memory.
    cudaFree(d_weights);
    cudaFree(d_distance);
    cudaFree(d_changed);
}

// bellman_ford_withThreads ==================================================================================
__global__ void bellman_ford_withThread_kernel(int *d_weights, int *d_distance, bool *d_changed) {

    for (int u = 0; u < VERTICES; u++) {
        for (int v = threadIdx.x; v < VERTICES; v += blockDim.x) {
            int weight = d_weights[u * VERTICES + v];
            if (weight < INF) {
                int new_distance = d_distance[u] + weight;
                if (new_distance < d_distance[v]) {
                    *d_changed = true;
                    d_distance[v] = new_distance;
                }
            }
        }
    }
}

void bellman_ford_withThread(int *weights, int *distance, int start, int blkdim, bool *has_negative_cycle) {

    int iter_num = 0;
    int *d_weights, *d_distance;
    bool *d_changed, h_changed;

    // initializing the distance array
    for (int i = 0; i < VERTICES; i++) {
        distance[i] = INF;
    }
    distance[start] = 0;

    // Allocate GPU memory for d_weights, d_distance, d_changed
    cudaMalloc(&d_weights, sizeof(int) * VERTICES * VERTICES);
    cudaMalloc(&d_distance, sizeof(int) * VERTICES);
    cudaMalloc(&d_changed, sizeof(bool));

    //Transfer the data from host to GPU.
    cudaMemcpy(d_weights, weights, sizeof(int) * VERTICES * VERTICES, cudaMemcpyHostToDevice);
    cudaMemcpy(d_distance, distance, sizeof(int) * VERTICES, cudaMemcpyHostToDevice);

    for (;;) {
        h_changed = false;
        cudaMemcpy(d_changed, &h_changed, sizeof(bool), cudaMemcpyHostToDevice);

        bellman_ford_withThread_kernel<<<1, blkdim>>>(d_weights, d_distance, d_changed);
        cudaDeviceSynchronize();
        cudaMemcpy(&h_changed, d_changed, sizeof(bool), cudaMemcpyDeviceToHost);

        iter_num++;
        if (iter_num >= VERTICES - 1) {
            *has_negative_cycle = true;
            break;
        }
        if (!h_changed) {
            break;
        }
    }
    if (!*has_negative_cycle) {
        // Copy the shortest path distances back to the host memory
        cudaMemcpy(distance, d_distance, sizeof(int) * VERTICES, cudaMemcpyDeviceToHost);
    }
    
    // Free up the GPU memory.
    cudaFree(d_weights);
    cudaFree(d_distance);
    cudaFree(d_changed);
}

// withBlocksThreads =========================================================================================
__global__ void bellman_ford_kernel(int *d_weights, int *d_distance, bool *d_changed) {
    int global_tid = blockDim.x * blockIdx.x + threadIdx.x;
    int elementSkip = blockDim.x * gridDim.x;

    if (global_tid < VERTICES){
        for (int u = 0; u < VERTICES; u++) {
            for (int v = global_tid; v < VERTICES; v += elementSkip) {
                int weight = d_weights[u * VERTICES + v];
                if (weight < INF) {
                    int new_distance = d_distance[u] + weight;
                    if (new_distance < d_distance[v]) {
                        *d_changed = true;
                        d_distance[v] = new_distance;
                    }
                }
            }
        }
    }
}

void bellman_ford(int *weights, int *distance, int start, int blkdim, bool *has_negative_cycle) {
    dim3 blocks((VERTICES + blkdim - 1) / blkdim);
    dim3 threads(blkdim);

    int iter_num = 0;
    int *d_weights, *d_distance;
    bool *d_changed, h_changed;

    // initializing the distance array
    for (int i = 0; i < VERTICES; i++) {
        distance[i] = INF;
    }
    distance[start] = 0;

    // Allocate GPU memory for d_weights, d_distance, d_changed
    cudaMalloc(&d_weights, sizeof(int) * VERTICES * VERTICES);
    cudaMalloc(&d_distance, sizeof(int) * VERTICES);
    cudaMalloc(&d_changed, sizeof(bool));

    //Transfer the data from host to GPU.
    cudaMemcpy(d_weights, weights, sizeof(int) * VERTICES * VERTICES, cudaMemcpyHostToDevice);
    cudaMemcpy(d_distance, distance, sizeof(int) * VERTICES, cudaMemcpyHostToDevice);

    for (;;) {
        h_changed = false;
        cudaMemcpy(d_changed, &h_changed, sizeof(bool), cudaMemcpyHostToDevice);

        bellman_ford_kernel<<<blocks, threads>>>(d_weights, d_distance, d_changed);
        cudaDeviceSynchronize();
        cudaMemcpy(&h_changed, d_changed, sizeof(bool), cudaMemcpyDeviceToHost);

        iter_num++;
        if (iter_num >= VERTICES - 1) {
            *has_negative_cycle = true;
            break;
        }
        if (!h_changed) {
            break;
        }
    }
    if (!*has_negative_cycle) {
        // Copy the shortest path distances back to the host memory
        cudaMemcpy(distance, d_distance, sizeof(int) * VERTICES, cudaMemcpyDeviceToHost);
    }
    
    // Free up the GPU memory.
    cudaFree(d_weights);
    cudaFree(d_distance);
    cudaFree(d_changed);
}

// ===========================================================================================================

int main(int argc, char **argv) {
    // make sure we pass blockPerGrid and threadsPerBlock
    assert(argv[1] != NULL);
    int blkdim = atoi(argv[1]);

    int n_edges;

    // initializing distance array
    int* distance = (int*)malloc(VERTICES * sizeof(int));
    // reading the adjacency matrix
    int* weights = (int*)malloc(VERTICES * VERTICES * sizeof(int));
    read_file("data/USA-road-NY.csv", weights, &n_edges);

    bool has_negative_cycle = false;
    double tstart, tend;
            
    printf("CUDA Specifications ==================\n");

    // recored the execution time
    cudaDeviceReset();
    tstart = gettime();
    bellman_ford_sequential(weights, distance, 0, &has_negative_cycle);
    cudaDeviceSynchronize();
    tend = gettime();

    printf("Sequential Implementation\n");
    printf("(blocks, threads):\t(1, 1)\n");
    printf("Exection time:\t\t%.6f sec\n\n", tend-tstart);

    // recored the execution time
    cudaDeviceReset();
    tstart = gettime();
    bellman_ford_withBlock(weights, distance, 0, &has_negative_cycle);
    cudaDeviceSynchronize();
    tend = gettime();

    printf("Block Parallel Implementation\n");
    printf("(blocks, threads):\t(%d, 1)\n", VERTICES);
    printf("Exection time:\t\t%.6f sec\n\n", tend-tstart);

    // recored the execution time
    cudaDeviceReset();
    tstart = gettime();
    bellman_ford_withThread(weights, distance, 0, blkdim, &has_negative_cycle);
    cudaDeviceSynchronize();
    tend = gettime();

    printf("Thread Implementation\n");
    printf("(blocks, threads):\t(1, %d)\n", blkdim);
    printf("Exection time:\t\t%.6f sec\n\n", tend-tstart);

    // recored the execution time
    cudaDeviceReset();
    tstart = gettime();
    bellman_ford(weights, distance, 0, blkdim, &has_negative_cycle);
    cudaDeviceSynchronize();
    tend = gettime();

    printf("Thread/Block Implementation\n");
    printf("(blocks, threads):\t(%d, %d)\n", ((VERTICES+blkdim-1)/blkdim), blkdim);
    printf("Exection time:\t\t%.6f sec\n\n", tend-tstart);

    save_results(distance, has_negative_cycle);

    return 0;
}
