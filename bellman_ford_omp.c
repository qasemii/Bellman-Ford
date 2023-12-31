// Dijkstra's Algorithm in C
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <omp.h>
#include <limits.h>
// #include "inc/algorithms.h"
// #include "inc/sort.h"
// #include "inc/config.h"


#define NUM_THREADS 1
#define VERTICES 983
#define INF 1000000


void print_result(bool has_negative_cycle, int *dist) {
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

float* Dijkstra(float** Graph, int n, int start) {
    float** cost;
    float* distance;
    int* visited;
    int* pred;
    int count, mindistance, nextnode, i, j;

    // Allocate memory for the cost matrix and other arrays
    cost = (float**)malloc(n * sizeof(float*));
    for (i = 0; i < n; i++) {
        cost[i] = (float*)malloc(n * sizeof(float));
    }
    
    distance = (float*)malloc(n * sizeof(float));
    pred = (int*)malloc(n * sizeof(int));
    visited = (int*)malloc(n * sizeof(int));

    // Creating cost matrix
    #pragma omp parallel for num_threads(NUM_THREADS) collapse(2)
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            if (Graph[i][j] == 0)
                cost[i][j] = INF;
            else
                cost[i][j] = Graph[i][j];
        }
    }

    #pragma omp parallel for num_threads(NUM_THREADS)
    for (i = 0; i < n; i++) {
        distance[i] = cost[start][i];
        pred[i] = start;
        visited[i] = 0;
    }

    distance[start] = 0;
    visited[start] = 1;
    count = 1;

    while (count < n - 1) {
        mindistance = INF;
        int local_mindistance = INF;
        int local_nextnode = 0;

        #pragma omp parallel for num_threads(NUM_THREADS)
        for (i = 0; i < n; i++) {
            if (distance[i] < local_mindistance && !visited[i]) {
                local_mindistance = distance[i];
                local_nextnode = i;
            }
        }

        #pragma omp critical
        {
            if (local_mindistance < mindistance) {
                mindistance = local_mindistance;
                nextnode = local_nextnode;
            }
        }

        visited[nextnode] = 1;

        #pragma omp parallel for num_threads(NUM_THREADS)
        for (i = 0; i < n; i++) {
            if (!visited[i])
                if (mindistance + cost[nextnode][i] < distance[i]) {
                    distance[i] = mindistance + cost[nextnode][i];
                    pred[i] = nextnode;
                }
        }
        count++;
    }

    // // Printing the distance
    // for (i = 0; i < n; i++)
    //     if (i != start) {
    //         printf("\nDistance from source to %d: %.3f", i, distance[i]);
    //     }

    // Free dynamically allocated memory
    for (i = 0; i < n; i++) {
        free(cost[i]);
    }
    free(cost);
    free(pred);
    free(visited);

    // free(distance);
    return distance;
}

int* BellmanFord(int** Graph, int n, int start) {
    int** cost;
    int* dist;

    // Allocate memory for the cost matrix and other arrays
    cost = (int**)malloc(n * sizeof(int*));
    for (int i = 0; i < n; i++) {
        cost[i] = (int*)malloc(n * sizeof(int));
    }

    // Creating cost matrix
    #pragma omp parallel for num_threads(NUM_THREADS) collapse(2)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (Graph[i][j] == 0)
                cost[i][j] = INF;
            else
                cost[i][j] = Graph[i][j];
        }
    }

    // step 3: initialize distances
    dist = (int*)malloc(n * sizeof(int));

    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        dist[i] = INF;
    }
    // root vertex always has distance 0
    dist[0] = 0;

    int local_start[NUM_THREADS], local_end[NUM_THREADS];
    bool *has_negative_cycle = false;

    // step 1: set openmp thread number
    omp_set_num_threads(NUM_THREADS);

    // step 2: find local task range
    int ave = n / NUM_THREADS;
    
    #pragma omp parallel for
    for (int i = 0; i < NUM_THREADS; i++) {
        local_start[i] = ave * i;
        local_end[i] = ave * (i + 1);
        if (i == NUM_THREADS - 1) {
            local_end[i] = n;
        }
    }

    int iter_num = 0;
    bool has_change;
    bool local_has_change[NUM_THREADS];
    #pragma omp parallel
    {
        int my_rank = omp_get_thread_num();
        // bellman-ford algorithm
        for (int iter = 0; iter < n - 1; iter++) {
            local_has_change[my_rank] = false;
            for (int u = 0; u < n; u++) {
                for (int v = local_start[my_rank]; v < local_end[my_rank]; v++) {
                    int weight = cost[u][v];
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
                for (int rank = 0; rank < NUM_THREADS; rank++) {
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
                int weight = cost[u][v];
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
    free(cost);
    return dist;
}

int main() {
    char line[256];
    double start, end;

    // Allocate memory for the matrix dynamically
    int** matrix = (int**)malloc(VERTICES * sizeof(int*));
    for (int i = 0; i < VERTICES; i++) {
        matrix[i] = (int*)malloc(VERTICES * sizeof(int));
    }

    // Initial the matrix with INFINITY for when there is no direct connection
    // We also set the diagonal elements to 0
    for (int i = 0; i < VERTICES; i++){
        for (int j = 0; j < VERTICES; j++){
            if (i != j){
                matrix[i][j] = INF; 
            }else{
                matrix[i][j] = 0;
            }
        }
    }

    // Open the CSV file
    FILE* file = fopen("data/london_temporal_at_23.csv", "r");
    if (file == NULL) {
        printf("Failed to open the CSV file.\n");
        return 1;
    }

    // Read each line in the CSV file and update the matrix
    int n_edges = 0;
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
            n_edges++;
            matrix[src_id][dest_id] = distance;
        }    
        
    }

    // Close the file
    fclose(file);

    int* distance = (int*)malloc(VERTICES * sizeof(int));
    start = omp_get_wtime();

    // all nodes to the others
    
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

    // one node to the others

    distance = BellmanFord(matrix, VERTICES, 0);
    end = omp_get_wtime();
    
    // // Printing the distance
    // for (int i = 0; i < VERTICES; i++)
    //     if (i != start) {
    //         printf("\nDistance from %d to %d: %d", 0, i, distance[i]);
    //     }

    free(distance);
    free(matrix);

    printf("Network Specifications----------\n");
    printf("Number of nodes:\t%d\n", VERTICES);
    printf("Number of edges:\t%d\n\n", n_edges);

    printf("OpenMP Specifications-----------\n");
    printf("Number of THREADS:\t%d\n", NUM_THREADS);
    printf("Execution time:\t\t%.6f sec\n\n", end-start);
    // printf("--------------------------------\n");
    print_result(false, distance);
    return 0;
}