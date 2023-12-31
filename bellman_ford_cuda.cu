#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <limits.h>
#include <assert.h>
#include <time.h>
// #include "hpc.h"

#define INF 1000000
#define VERTICES 983

double gettime(void){
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts );
    return (ts.tv_sec + (double)ts.tv_nsec / 1e9);
}

void read_file(const char* filename, int* weights) {
    // Initialize the matrix with INT_MAX and 0 for diagonals
    for (int i = 0; i < VERTICES; i++) {
        for (int j = 0; j < VERTICES; j++) {
            if (i != j) {
                weights[i * VERTICES + j] = INT_MAX;
            } else {
                weights[i * VERTICES + j] = 0;
            }
        }
    }

    // Open the CSV file
    FILE* file = fopen(filename, "r");

    // Read each line in the CSV file and update the matrix
    char line[256];
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
            weights[src_id * VERTICES + dest_id] = distance;
        }
    }
    fclose(file);
}

void save_results(int *dist, bool has_negative_cycle) {
    FILE *outputf = fopen("cuda_output.txt", "w");
    if (!has_negative_cycle) {
        for (int i = 0; i < VERTICES; i++) {
            if (dist[i] > INT_MAX)
                dist[i] = INT_MAX;
            fprintf(outputf, "%d\n", dist[i]);
        }
        fflush(outputf);
    } else {
        fprintf(outputf, "Negative cycle detected!\n");
    }
    fclose(outputf);
}

__global__ void bellman_ford_kernel(int *d_weights, int *d_distance, int n, bool *d_has_next) {
    int global_tid = blockDim.x * blockIdx.x + threadIdx.x;
    int elementSkip = blockDim.x * gridDim.x;

    if (global_tid < n){
        for (int u = 0; u < n; u++) {
            for (int v = global_tid; v < n; v += elementSkip) {
                int weight = d_weights[u * n + v];
                if (weight < INF) {
                    int new_dist = d_distance[u] + weight;
                    if (new_dist < d_distance[v]) {
                        *d_has_next = true;
                        d_distance[v] = new_dist;
                    }
                }
            }
        }
    }
}

void bellman_ford(int *weights, int *distance, int start, int n, int blocksPerGrid, int threadsPerBlock, bool *has_negative_cycle) {
    dim3 blocks(blocksPerGrid);
    dim3 threads(threadsPerBlock);

    int iter_num = 0;
    int *d_weights, *d_distance;
    bool *d_has_next, h_has_next;

    // initializing the distance array
    for (int i = 0; i < n; i++) {
        distance[i] = INF;
    }
    distance[start] = 0;

    // Allocate GPU memory for d_weights, d_distance, d_has_next
    cudaMalloc(&d_weights, sizeof(int) * n * n);
    cudaMalloc(&d_distance, sizeof(int) * n);
    cudaMalloc(&d_has_next, sizeof(bool));

    //Transfer the data from host to GPU.
    cudaMemcpy(d_weights, weights, sizeof(int) * n * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_distance, distance, sizeof(int) * n, cudaMemcpyHostToDevice);

    for (;;) {
        h_has_next = false;
        cudaMemcpy(d_has_next, &h_has_next, sizeof(bool), cudaMemcpyHostToDevice);

        bellman_ford_kernel<<<blocks, threads>>>(d_weights, d_distance, n, d_has_next);
        cudaDeviceSynchronize();
        cudaMemcpy(&h_has_next, d_has_next, sizeof(bool), cudaMemcpyDeviceToHost);

        iter_num++;
        if (iter_num >= n - 1) {
            *has_negative_cycle = true;
            break;
        }
        if (!h_has_next) {
            break;
        }
    }
    if (!*has_negative_cycle) {
        // Copy the shortest path distances back to the host memory
        cudaMemcpy(distance, d_distance, sizeof(int) * n, cudaMemcpyDeviceToHost);
    }
    
    // Free up the GPU memory.
    cudaFree(d_weights);
    cudaFree(d_distance);
    cudaFree(d_has_next);
}

int main(int argc, char **argv) {
    // make sure we pass blockPerGrid and threadsPerBlock
    assert(argv[1] != NULL && argv[2]!=NULL);
    int blocksPerGrid = atoi(argv[1]);
    int threadsPerBlock = atoi(argv[2]);
    
    // reading the adjacency matrix
    int* weights = (int*)malloc(VERTICES * VERTICES * sizeof(int));
    read_file("data/london_temporal_at_23.csv", weights);

    // initializing distance array
    int* distance = (int*)malloc(VERTICES * sizeof(int));

    bool* has_negative_cycle = (bool*)malloc(VERTICES * sizeof(bool));
    has_negative_cycle = false;

    double tstart, tend;

    // recored the execution time
    cudaDeviceReset();
    tstart = gettime();
    bellman_ford(weights, distance, 0, VERTICES, blocksPerGrid, threadsPerBlock, has_negative_cycle);
    cudaDeviceSynchronize();
    tend = gettime();

    printf("CUDA Specifications-------------\n");
    printf("blockPerGrid:\t\t%d\n", blocksPerGrid);
    printf("threadsPerBlock:\t%d\n", threadsPerBlock);
    printf("Exection time:\t\t%.6f sec\n\n", tend-tstart);

    save_results(distance, has_negative_cycle);

    return 0;
}
