#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <omp.h>
#include <limits.h>
#include "inc/algorithms.h"
#include "inc/sort.h"

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
    // Printing the distance
    for (int i = 0; i < VERTICES; i++)
        if (i != start) {
            printf("\nDistance from %d to %d: %.3f", 0, i, distance[i]);
        }
    end = omp_get_wtime();

    free(distance);
    free(matrix);

    printf("\nExe time: %e sec\n", end-start);
    return 0;
}
