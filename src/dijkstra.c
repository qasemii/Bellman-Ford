// Dijkstra's Algorithm in C
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <omp.h>
#include <limits.h>
#include "../inc/dijkstra.h"



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

    // #pragma omp parallel for num_threads(NUM_THREADS)
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

        // #pragma omp critical
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
