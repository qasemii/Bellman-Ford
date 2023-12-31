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
#define MAX_FIELD_SIZE 1024
#define VERTICES 983
#define MAX 983
#define INFINITY 9999


void print_result(bool has_negative_cycle, int *dist) {
    FILE *outputf = fopen("omp_output.txt", "w");
    if (!has_negative_cycle) {
        for (int i = 0; i < VERTICES; i++) {
            if (dist[i] > INF)
                dist[i] = INF;
            fprintf("%d\n", dist[i]);
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
                cost[i][j] = INFINITY;
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
        mindistance = INFINITY;
        int local_mindistance = INFINITY;
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

float* BellmanFord(float** Graph, int n, int start) {
    float** cost;
    float* dist;

    // Allocate memory for the cost matrix and other arrays
    cost = (float**)malloc(n * sizeof(float*));
    for (int i = 0; i < n; i++) {
        cost[i] = (float*)malloc(n * sizeof(float));
    }

    // Creating cost matrix
    #pragma omp parallel for num_threads(NUM_THREADS) collapse(2)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (Graph[i][j] == 0)
                cost[i][j] = INFINITY;
            else
                cost[i][j] = Graph[i][j];
        }
    }

    // step 3: initialize distances
    dist = (float*)malloc(n * sizeof(float));

    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        dist[i] = INFINITY;
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
                    if (weight < INFINITY) {
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
                if (weight < INFINITY) {
                    if (dist[u] + weight < dist[v]) { // if we can relax one more step, then we find a negative cycle
                        has_change = true;
                        ;
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
    int max_src_id = VERTICES;
    int max_dest_id = VERTICES;
    double start, end;


    // Allocate memory for the matrix dynamically
    float** matrix = (float**)malloc(VERTICES * sizeof(float*));
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
    FILE* file = fopen("data/london_temporal_at_23.csv", "r");
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

    float* distance = (float*)malloc(VERTICES * sizeof(float));
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
    
    // Printing the distance
    for (int i = 0; i < VERTICES; i++)
        if (i != start) {
            printf("\nDistance from %d to %d: %.3f", 0, i, distance[i]);
        }

    free(distance);
    free(matrix);

    printf("Network Specifications----------\n");
    printf("Number of nodes:\t%d\n", VERTICES);
    printf("Number of edges:\t%d\n", n_edges);
    printf("OpenMP Specifications-----------\n");
    printf('Number of THREADS:\t%d\n', NUM_THREADS)
    printf("Exe time:\t%.6f sec\n", end-start);
    printf("--------------------------------\n");
    print_result(false, int *dist)
    return 0;
}