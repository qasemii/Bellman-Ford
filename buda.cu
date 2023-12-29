
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <omp.h>
#include <limits.h>
#include "../inc/algorithms.h"
#include "../inc/config.h"

#define BLOCK_SIZE 256

// CUDA kernel to perform distance updates in parallel
__global__ void bellmanFordKernel(int n, float** Graph, int* dist, bool* hasChange) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < n) {
        for (int v = 0; v < n; v++) {
            int weight = Graph[tid][v];
            if (dist[tid] + weight < dist[v]) {
                dist[v] = dist[tid] + weight;
                *hasChange = true;
            }
        }
    }
}

// Function to perform the Bellman-Ford algorithm using CUDA
void bellmanFordCUDA(float** graph, int n, int start) {
    int* d_graph;
    int* d_dist;
    bool* d_hasChange;

    // Allocate device memory
    cudaMalloc((void**)&d_graph, n * n * sizeof(int));
    cudaMalloc((void**)&d_dist, n * sizeof(int));
    cudaMalloc((void**)&d_hasChange, sizeof(bool));

    // Copy graph data to device
    cudaMemcpy(d_graph, graph, n * n * sizeof(int), cudaMemcpyHostToDevice);

    // Initialize distances on the device
    int* dist = (int*)malloc(n * sizeof(int));
    for (int i = 0; i < n; i++)
        dist[i] = (i == start) ? 0 : INT_MAX;

    cudaMemcpy(d_dist, dist, n * sizeof(int), cudaMemcpyHostToDevice);

    // Initialize the host variable hasChange
    bool hasChange = true;
    bool* h_hasChange = (bool*)malloc(sizeof(bool));
    *h_hasChange = true;

    // Main loop of Bellman-Ford
    while (hasChange) {
        hasChange = false;
        cudaMemcpy(d_hasChange, h_hasChange, sizeof(bool), cudaMemcpyHostToDevice);

        // Launch the CUDA kernel
        bellmanFordKernel<<<(n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(n, d_graph, d_dist, d_hasChange);

        // Copy hasChange back to host
        cudaMemcpy(h_hasChange, d_hasChange, sizeof(bool), cudaMemcpyDeviceToHost);

        hasChange = *h_hasChange;
    }

    // Copy the final distances back to host
    cudaMemcpy(dist, d_dist, n * sizeof(int), cudaMemcpyDeviceToHost);

    // Print the distances
    printf("Vertex   Distance from Source\n");
    for (int i = 0; i < n; i++)
        printf("%d \t\t %d\n", i, dist[i]);

    // Free device memory
    cudaFree(d_graph);
    cudaFree(d_dist);
    cudaFree(d_hasChange);
}

int main() {
    FILE* file;
    char line[256];
    int max_src_id = VERTICES;
    int max_dest_id = VERTICES;
    float** matrix;

    double start, end;


    // Allocate memory for the matrix dynamically
    matrix = (float**)malloc(VERTICES * sizeof(float*));
    for (int i = 0; i < VERTICES; i++) {
        matrix[i] = (float*)malloc(VERTICES * sizeof(float));
    }

    // Initial the matrix with INFINITY for when there is no direct connection
    // We also set the diagonal elements to 0
    for (int i = 0; i < VERTICES; i++){
        for (int j = 0; j < VERTICES; j++){
            if (i != j){
                matrix[i][j] = INFINITY; 
            }else{
                matrix[i][j] = 0;
            }
        }
    }

    // Open the CSV file
    file = fopen("data/london_temporal_at_23.csv", "r");
    if (file == NULL) {
        printf("Failed to open the CSV file.\n");
        return 1;
    }

    // Read each line in the CSV file and update the matrix
    int n_edges = 0;
    while (fgets(line, sizeof(line), file)) {
        char* field;
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
            n_edges++;
            matrix[src_id][dest_id] = distance;
        }    
        
    }

    // Close the file
    fclose(file);
    printf("Network Specifications");
    printf("\nNumber of nodes: %d", VERTICES);
    printf("\nNumber of edges: %d", n_edges);
    printf("\n----------------------------------------------");


    bellmanFordCUDA(matrix, VERTICES, 0);

    return 0;
}
