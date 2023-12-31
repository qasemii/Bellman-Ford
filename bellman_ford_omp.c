#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <omp.h>
#include <limits.h>
#include <assert.h>


#define INF 999999
#define VERTICES 983


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

void save_results(int *dist, bool has_negative_cycle) {
    FILE *outputf = fopen("omp_output.txt", "w");
    if (!has_negative_cycle) {
        for (int i = 0; i < VERTICES; i++) {
            if (dist[i] > INF)
                dist[i] = INF;
            fprintf(outputf, "%d\n", dist[i]);
        }
        fflush(outputf);
    } else {
        fprintf(outputf, "Negative cycle detected!\n");
    }
    fclose(outputf);
}

void BellmanFord(int* weights, int* distance, int start, int n, int n_threads, bool* has_negative_cycle) {

    int local_start[n_threads], local_end[n_threads];
    
    //find local task range
    int ave = n / n_threads;
    
    //set openmp thread number
    omp_set_num_threads(n_threads);

    // initializing the distance array
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        distance[i] = INF;
    }
    distance[start] = 0;

    #pragma omp parallel for
    for (int i = 0; i < n_threads; i++) {
        local_start[i] = ave * i;
        local_end[i] = ave * (i + 1);
    }
    local_end[n_threads-1] = n;

    int iter_num = 0;
    bool has_change;
    bool local_has_change[n_threads];
    #pragma omp parallel
    {
        int my_rank = omp_get_thread_num();
        for (int iter = 0; iter < n - 1; iter++) {
            local_has_change[my_rank] = false;
            for (int u = 0; u < n; u++) {
                for (int v = local_start[my_rank]; v < local_end[my_rank]; v++) {
                    int weight = weights[u * n + v];
                    if (weight < INF) {
                        int new_dis = distance[u] + weight;
                        if (new_dis < distance[v]) {
                            local_has_change[my_rank] = true;
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
                has_change = false;
                for (int rank = 0; rank < n_threads; rank++) {
                    has_change |= local_has_change[rank];
                }
            }
            if (!has_change) {
                break;
            }
        }
    }

    // check negative cycles
    if (iter_num == n - 1) {
        has_change = false;
        for (int u = 0; u < n; u++) {
            #pragma omp parallel for reduction(| : has_change)
            for (int v = 0; v < n; v++) {
                int weight = weights[u * n + v];
                if (weight < INF) {
                    if (distance[u] + weight < distance[v]) { // if we can relax one more step, then we find a negative cycle
                        has_change = true;
                    }
                }
            }
        }
        *has_negative_cycle = has_change;
    }

    free(weights);
}

int main(int argc, char **argv) {
    // make sure we pass number of threads (N_THREADS)
    assert(argv[1] != NULL);
    int n_threads = atoi(argv[1]);

    int n_edges;

    // reading the adjacency matrix
    int* weights = (int*)malloc(VERTICES * VERTICES * sizeof(int));
    read_file("data/london_temporal_at_23.csv", weights, &n_edges);
    // initializing distance array
    int* distance = (int*)malloc(VERTICES * sizeof(int));

    bool has_negative_cycle = false;

    double tstart, tend;

    // recored the execution time
    tstart = omp_get_wtime();
    BellmanFord(weights, distance, 0, VERTICES, n_threads, &has_negative_cycle);
    tend = omp_get_wtime();

    printf("Network Specifications----------\n");
    printf("Number of nodes:\t%d\n", VERTICES);
    printf("Number of edges:\t%d\n\n", n_edges);
    printf("OpenMP Specifications-----------\n");
    printf("Number of THREADS:\t%d\n", n_threads);
    printf("Execution time:\t\t%.6f sec\n\n", tend-tstart);

    save_results(distance, false);

    return 0;
}