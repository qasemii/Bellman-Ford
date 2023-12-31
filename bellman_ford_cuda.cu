#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <limits.h>
#include <assert.h>
#include <time.h>
// #include "hpc.h"

#define INF 1000000
#define VERTICES 983

double gettime(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts );
    return (ts.tv_sec + (double)ts.tv_nsec / 1e9);
}

void abort_with_error_message(const char *msg) {
    fprintf(stderr, "%s\n", msg);
    exit(1);
}

int* read_file(const char* filename) {
    char line[256];
    int* mat = (int*)malloc(VERTICES * VERTICES * sizeof(int));

    // Initialize the matrix with INFINITY
    for (int i = 0; i < VERTICES; i++) {
        for (int j = 0; j < VERTICES; j++) {
            if (i != j) {
                mat[i * VERTICES + j] = INF;
            } else {
                mat[i * VERTICES + j] = 0;
            }
        }
    }

    // Open the CSV file
    FILE* file = fopen(filename, "r");

    // Read each line in the CSV file and update the matrix
    while (fgets(line, sizeof(line), file)) {
        char* token;
        char* rest = line;
        int src_id, dest_id;
        float distance;

        // Tokenize the line based on the comma delimiter
        token = strtok_r(rest, ",", &rest);
        src_id = atoi(token);

        token = strtok_r(rest, ",", &rest);
        dest_id = atoi(token);

        token = strtok_r(rest, ",", &rest);
        distance = atof(token);

        // Update the matrix with the distance value
        if (src_id < VERTICES && dest_id < VERTICES) {
            mat[src_id * VERTICES + dest_id] = distance;
        }
    }
    fclose(file);
    return mat;
}

void save(bool has_negative_cycle, int *dist) {
    FILE *outputf = fopen("cuda_output.txt", "w");
    if (!has_negative_cycle) {
        for (int i = 0; i < VERTICES; i++) {
            if (dist[i] > INF)
                dist[i] = INF;
            fprintf(outputf, "%d\n", dist[i]);
        }
        fflush(outputf);
    } else {
        printf("FOUND NEGATIVE CYCLE!\n");
    }
    fclose(outputf);
}

__global__ void bellman_ford_one_iter(int n, int *d_mat, int *d_dist, bool *d_has_next, int iter_num) {
    int global_tid = blockDim.x * blockIdx.x + threadIdx.x;
    int elementSkip = blockDim.x * gridDim.x;

    if (global_tid >= n)
        return;
    for (int u = 0; u < n; u++) {
        for (int v = global_tid; v < n; v += elementSkip) {
            int weight = d_mat[u * n + v];
            if (weight < INF) {
                int new_dist = d_dist[u] + weight;
                if (new_dist < d_dist[v]) {
                    *d_has_next = true;
                    d_dist[v] = new_dist;
                }
            }
        }
    }
}

void bellman_ford(int blocksPerGrid, int threadsPerBlock, int n, int *mat, int *dist, bool *has_negative_cycle) {
    dim3 blocks(blocksPerGrid);
    dim3 threads(threadsPerBlock);

    int iter_num = 0;
    int *d_mat, *d_dist;
    bool *d_has_next, h_has_next;

    cudaMalloc(&d_mat, sizeof(int) * n * n);
    cudaMalloc(&d_dist, sizeof(int) * n);
    cudaMalloc(&d_has_next, sizeof(bool));

    *has_negative_cycle = false;

    for (int i = 0; i < n; i++) {
        dist[i] = INF;
    }

    dist[0] = 0;
    cudaMemcpy(d_mat, mat, sizeof(int) * n * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_dist, dist, sizeof(int) * n, cudaMemcpyHostToDevice);

    for (;;) {
        h_has_next = false;
        cudaMemcpy(d_has_next, &h_has_next, sizeof(bool), cudaMemcpyHostToDevice);

        bellman_ford_one_iter<<<blocks, threads>>>(n, d_mat, d_dist, d_has_next, iter_num);
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
        cudaMemcpy(dist, d_dist, sizeof(int) * n, cudaMemcpyDeviceToHost);
    }

    cudaFree(d_mat);
    cudaFree(d_dist);
    cudaFree(d_has_next);
}

int main(int argc, char **argv) {
    if (argc <= 2) {
        abort_with_error_message("blocksPerGrid or threadsPerBlock WAS NOT FOUND!");
    }
    int blockPerGrid = atoi(argv[1]);
    int threadsPerBlock = atoi(argv[2]);
    
    bool has_negative_cycle = false;
    
    int* mat = read_file("data/london_temporal_at_23.csv");
    int dist[VERTICES];
    memset(dist, 0, sizeof(dist));

    // time counter
    double tstart, tend;

    cudaDeviceReset();

    tstart = gettime();
    bellman_ford(blockPerGrid, threadsPerBlock, VERTICES, mat, dist, &has_negative_cycle);
    cudaDeviceSynchronize();
    tend = gettime();

    printf("CUDA Specifications-------------\n");
    printf("blockPerGrid:\t\t%d\n", blockPerGrid);
    printf("threadsPerBlock:\t%d\n", threadsPerBlock);
    printf("Exection time:\t\t%.6f sec\n\n", tend-tstart);

    save(has_negative_cycle, dist);
    return 0;
}
