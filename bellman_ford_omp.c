#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <omp.h>
#include <limits.h>


#define VERTICES 983
#define INF 1000000

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

void save_results(bool has_negative_cycle, int *dist) {
    FILE *outputf = fopen("omp_output.txt", "w");
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

// (int p, int n, int *mat, int *dist, bool *has_negative_cycle)
void BellmanFord(int n_threads, int* mat, int n, int start, int* dist) {

    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        dist[i] = INF;
    }
    // root vertex always has distance 0
    dist[0] = 0;


    int local_start[n_threads], local_end[n_threads];
    bool *has_negative_cycle = false;

    // step 1: set openmp thread number
    omp_set_num_threads(n_threads);

    // step 2: find local task range
    int ave = n / n_threads;
    
    #pragma omp parallel for
    for (int i = 0; i < n_threads; i++) {
        local_start[i] = ave * i;
        local_end[i] = ave * (i + 1);
        if (i == n_threads - 1) {
            local_end[i] = n;
        }
    }

    int iter_num = 0;
    bool has_change;
    bool local_has_change[n_threads];
    #pragma omp parallel
    {
        int my_rank = omp_get_thread_num();
        // bellman-ford algorithm
        for (int iter = 0; iter < n - 1; iter++) {
            local_has_change[my_rank] = false;
            for (int u = 0; u < n; u++) {
                for (int v = local_start[my_rank]; v < local_end[my_rank]; v++) {
                    int weight = mat[u * n + v];
                    if (weight < INF) {
                        int new_dis = dist[u] + weight;
                        if (new_dis < dist[v]) {
                            local_has_change[my_rank] = true;
                            dist[v] = new_dis;
                        }
                    }
                }
            }
            #pragma omp barrier
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

    // do one more iteration to check negative cycles
    if (iter_num == n - 1) {
        has_change = false;
        for (int u = 0; u < n; u++) {
            #pragma omp parallel for reduction(| : has_change)
            for (int v = 0; v < n; v++) {
                int weight = mat[u * n + v];
                if (weight < INF) {
                    if (dist[u] + weight < dist[v]) { // if we can relax one more step, then we find a negative cycle
                        has_change = true;
                    }
                }
            }
        }
        *has_negative_cycle = has_change;
    }

    // step 4: free memory (if any)
    free(mat);
}

int main(int argc, char **argv) {
    if (argc <= 1) {
        abort_with_error_message("N_THREADS is not defined");
    }
    int N_THREADS = atoi(argv[1]);

    int* mat = read_file("data/london_temporal_at_23.csv");
    int* dist = (int*)malloc(VERTICES * sizeof(int));

    double tstart, tend;
    tstart = omp_get_wtime();

    // all nodes to the others =====================================
    // // #pragma omp parallel num_threads(NUM_THREADS) private(distance)
    // for (int u = 0; u < VERTICES; u++){
    //     // int u = 0;
    //     distance = BellmanFord(matrix, VERTICES, u);
    //     merge_sort(distance, 0, VERTICES-1);
    //     // Printing the distance
    //     for (int i = 0; i < VERTICES; i++)
    //         if (i != start) {
    //             printf("\nDistance from %d to %d: %.3f", u, i, distance[i]);
    //         }
    //     free(distance);
    // }
    // // Free dynamically allocated memory for the Graph
    // for (int i = 0; i < VERTICES; i++) {
    //     free(matrix[i]);
    // }

    // one node to the others =====================================
    BellmanFord(N_THREADS, mat, VERTICES, 0, dist);
    tend = omp_get_wtime();

    printf("Network Specifications----------\n");
    printf("Number of nodes:\t%d\n", VERTICES);
    // printf("Number of edges:\t%d\n\n", n_edges);

    printf("OpenMP Specifications-----------\n");
    printf("Number of THREADS:\t%d\n", N_THREADS);
    printf("Execution time:\t\t%.6f sec\n\n", tend-tstart);

    save_results(false, dist);


    free(dist);
    free(mat);
    return 0;
}