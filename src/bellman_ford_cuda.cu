/*
 * This is a CUDA version of bellman_ford algorithm
 * Compile: nvcc -std=c++11 -arch=sm_52 -o cuda_bellman_ford cuda_bellman_ford.cu
 * Run: ./cuda_bellman_ford <input file> <number of blocks per grid> <number of threads per block>, you will find the output file 'output.txt'
 * */

#include <string>
#include <cassert>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <iomanip>
#include <cstring>
#include <sys/time.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <omp.h>
#include <limits.h>
#include "../inc/algorithms.h"
#include "../inc/config.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

using std::string;
using std::cout;
using std::endl;


/*
 * This is a CHECK function to check CUDA calls
 */
#define CHECK(call)                                                            \
		{                                                                              \
	const cudaError_t error = call;                                            \
	if (error != cudaSuccess)                                                  \
	{                                                                          \
		fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                 \
		fprintf(stderr, "code: %d, reason: %s\n", error,                       \
				cudaGetErrorString(error));                                    \
				exit(1);                                                               \
	}                                                                          \
		}


/**
 * utils is a namespace for utility functions
 * including I/O (read input file and print results) and matrix dimension convert(2D->1D) function
 */
namespace utils {
int N; //number of vertices
int *mat; // the adjacency matrix

void abort_with_error_message(string msg) {
	std::cerr << msg << endl;
	abort();
}

//translate 2-dimension coordinate to 1-dimension
int convert_dimension_2D_1D(int x, int y, int n) {
	return x * n + y;
}

int read_file(string filename) {
	std::ifstream inputf(filename, std::ifstream::in);
	if (!inputf.good()) {
		abort_with_error_message("ERROR OCCURRED WHILE READING INPUT FILE");
	}
	inputf >> N;
	//input matrix should be smaller than 20MB * 20MB (400MB, we don't have too much memory for multi-processors)
	assert(N < (1024 * 1024 * 20));
	mat = (int *) malloc(N * N * sizeof(int));
	for (int i = 0; i < N; i++)
		for (int j = 0; j < N; j++) {
			inputf >> mat[convert_dimension_2D_1D(i, j, N)];
		}
	return 0;
}

int print_result(bool has_negative_cycle, int *dist) {
	std::ofstream outputf("output.txt", std::ofstream::out);
	if (!has_negative_cycle) {
		for (int i = 0; i < N; i++) {
			if (dist[i] > INFINITY)
				dist[i] = INFINITY;
			outputf << dist[i] << '\n';
		}
		outputf.flush();
	} else {
		outputf << "FOUND NEGATIVE CYCLE!" << endl;
	}
	outputf.close();
	return 0;
}
}//namespace utils


__global__ void bellmanFordKernel(int n, int *d_mat, int *d_dist, bool *d_has_next, int iter_num){
	int global_tid = blockDim.x * blockIdx.x + threadIdx.x;
	int elementSkip = blockDim.x * gridDim.x;

	if(global_tid < n) {
        for(int u = 0 ; u < n ; u ++){
            for(int v = global_tid; v < n; v+= elementSkip){
                int weight = d_mat[u * n + v];
                if(weight < INFINITY){
                    int new_dist = d_dist[u] + weight;
                    if(new_dist < d_dist[v]){
                        d_dist[v] = new_dist;
                        *d_has_next = true;
                    }
                }
            }
        }
    }

}

/**
 * Bellman-Ford algorithm. Find the shortest path from vertex 0 to other vertices.
 * @param blockPerGrid number of blocks per grid
 * @param threadsPerBlock number of threads per block
 * @param n input size
 * @param *mat input adjacency matrix
 * @param *dist distance array
 * @param *has_negative_cycle a bool variable to recode if there are negative cycles
 */
float* bellman_ford(float** Graph, int n, int start, int blocksPerGrid, int threadsPerBlock) {
    float** cost;
    float* dist;

	dim3 blocks(blocksPerGrid);
	dim3 threads(threadsPerBlock);

	int iter_num = 0;
	int *d_mat, *d_dist;
	bool *d_has_next, h_has_next;

	cudaMalloc(&d_mat, sizeof(int) * n * n);
	cudaMalloc(&d_dist, sizeof(int) *n);
	cudaMalloc(&d_has_next, sizeof(bool));

	*has_negative_cycle = false;

	for(int i = 0 ; i < n; i ++){
		dist[i] = INFINITY;
	}

	dist[0] = 0;
	cudaMemcpy(d_mat, mat, sizeof(int) * n * n, cudaMemcpyHostToDevice);
	cudaMemcpy(d_dist, dist, sizeof(int) * n, cudaMemcpyHostToDevice);

	for(;;){
		h_has_next = false;
		cudaMemcpy(d_has_next, &h_has_next, sizeof(bool), cudaMemcpyHostToDevice);

		bellmanFordKernel<<<blocks, threads>>>(n, d_mat, d_dist, d_has_next, iter_num);
		CHECK(cudaDeviceSynchronize());
		cudaMemcpy(&h_has_next, d_has_next, sizeof(bool), cudaMemcpyDeviceToHost);

		iter_num++;
		if(iter_num >= n-1){
			*has_negative_cycle = true;
			break;
		}
		if(!h_has_next){
			break;
		}

	}
	if(! *has_negative_cycle){
		cudaMemcpy(dist, d_dist, sizeof(int) * n, cudaMemcpyDeviceToHost);
	}

	cudaFree(d_mat);
	cudaFree(d_dist);
	cudaFree(d_has_next);
}

int main(int argc, char **argv) {
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


	float* distance = (float*)malloc(VERTICES * sizeof(float));

	//time counter
	timeval start_wall_time_t, end_wall_time_t;
	float ms_wall;
	cudaDeviceReset();
	//start timer
	gettimeofday(&start_wall_time_t, nullptr);
	//bellman-ford algorithm
	bellman_ford(blockPerGrid, threadsPerBlock, utils::N, utils::mat, dist, &has_negative_cycle);
	// CHECK(cudaDeviceSynchronize());
	//end timer
	gettimeofday(&end_wall_time_t, nullptr);
	ms_wall = ((end_wall_time_t.tv_sec - start_wall_time_t.tv_sec) * 1000 * 1000
			+ end_wall_time_t.tv_usec - start_wall_time_t.tv_usec) / 1000.0;

    
	utils::print_result(has_negative_cycle, dist);
	
    free(dist);
	free(matrix);

    printf("\nExe time: %e sec\n", (ms_wall/1000.0) << endl);

	return 0;
}