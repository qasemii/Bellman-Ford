#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <omp.h>
#include <limits.h>
#include <assert.h>


#define INF 9999999
#define VERTICES 5000 //total vertices 264,346
#define START 2978 //this is the node with maximum outgoing edges
#define N_THREADS 2


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
    *n_edges = 0;
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
            (*n_edges)++;
            weights[src_id * VERTICES + dest_id] = distance;
        }
    }
    fclose(file);
}

void save_results(int *dist) {
    FILE *outputf = fopen("omp_output.txt", "w");
    for (int i = 0; i < VERTICES; i++) {
        if (dist[i] > INF)
            dist[i] = INF;
        fprintf(outputf, "%d\n", dist[i]);
    }
    fflush(outputf);
    fclose(outputf);
}

void bellman_ford(int* weights, int* distance, int start, int n_theads) {

    int local_start[n_threads], local_end[n_threads];
    
    //find local task range
    int ave = VERTICES / n_threads;
    
    //set openmp thread number
    omp_set_num_threads(n_threads);

    // initializing the distance array
    #pragma omp parallel for
    for (int i = 0; i < VERTICES; i++) {
        distance[i] = INF;
    }
    distance[start] = 0;

    #pragma omp parallel for
    for (int i = 0; i < n_threads; i++) {
        local_start[i] = ave * i;
        local_end[i] = ave * (i + 1);
    }
    local_end[n_threads-1] = VERTICES;

    int iter_num = 0;
    bool changed;
    bool local_changed[n_threads];
    #pragma omp parallel
    {
        int my_rank = omp_get_thread_num();
        for (int iter = 0; iter < VERTICES - 1; iter++) {
            local_changed[my_rank] = false;
            for (int u = 0; u < VERTICES; u++) {
                for (int v = local_start[my_rank]; v < local_end[my_rank]; v++) {
                    int weight = weights[u * VERTICES + v];
                    if (weight < INF) {
                        int new_dis = distance[u] + weight;
                        if (new_dis < distance[v]) {
                            local_changed[my_rank] = true;
                            distance[v] = new_dis;
                        }
                    }
                }
            }
            // wait for all threads to finish
            #pragma omp barrier

            // single thread execution
            #pragma omp single
            {
                iter_num++;
                changed = false;
                for (int rank = 0; rank < n_threads; rank++) {
                    changed |= local_changed[rank];
                }
            }
            if (!changed) {
                break;
            }
        }
    }

    free(weights);
}

int main() {

    int n_edges;

    // initializing distance array
    int* distance = (int*)malloc(VERTICES * sizeof(int));
    // reading the adjacency matrix
    int* weights = (int*)malloc(VERTICES * VERTICES * sizeof(int));
    read_file("data/USA-road-NY.csv", weights, &n_edges);

    double tstart, tend;

    // recored the execution time
    tstart = omp_get_wtime();
    bellman_ford(weights, distance, START, n_threads);
    tend = omp_get_wtime();

    printf("Network Specifications ===============\n");
    printf("Number of nodes:\t%d\n", VERTICES);
    printf("Number of edges:\t%d\n", n_edges);
    // printf("Density of graph:\t%.6f\n\n", (float)n_edges/(VERTICES*(VERTICES-1)));

    printf("OpenMP Specifications ================\n");
    printf("Sequential Implementation\n");
    printf("Number of THREADS:\t%d\n", 1);
    printf("Execution time:\t\t%.6f sec\n\n", tend-tstart);

    printf("Multicore Implementation\n");
    printf("Number of THREADS:\t%d\n", N_THREADS);
    printf("Execution time:\t\t%.6f sec\n\n", tend-tstart);

    save_results(distance);

    return 0;
}