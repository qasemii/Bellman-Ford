#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#include "../inc/config.h"
#include "../inc/sort.h"

void merge(float* arr, int left, int mid, int right) {
    int i, j, k;
    int n1 = mid - left + 1;
    int n2 = right - mid;

    // Create temporary arrays to store the left and right halves
    int L[n1], R[n2];

    // Copy data to the temporary arrays
    for (i = 0; i < n1; i++) {
        L[i] = arr[left + i];
    }
    for (j = 0; j < n2; j++) {
        R[j] = arr[mid + 1 + j];
    }

    // Merge the temporary arrays back into arr[left..right]
    i = 0; // Initial index of left subarray
    j = 0; // Initial index of right subarray
    k = left; // Initial index of merged subarray

    while (i < n1 && j < n2) {
        if (L[i] <= R[j]) {
            arr[k] = L[i];
            i++;
        } else {
            arr[k] = R[j];
            j++;
        }
        k++;
    }

    // Copy the remaining elements of L[], if any
    while (i < n1) {
        arr[k] = L[i];
        i++;
        k++;
    }

    // Copy the remaining elements of R[], if any
    while (j < n2) {
        arr[k] = R[j];
        j++;
        k++;
    }
}

void merge_sort(float* arr, int left, int right) {
    if (left < right) {
        int mid = left + (right - left) / 2;

        // Parallelize the sorting of the left and right halves using OpenMP tasks
        #pragma omp parallel num_threads(NUM_THREADS)
        {
            #pragma omp single nowait
            {
                #pragma omp task
                merge_sort(arr, left, mid);
            }

            #pragma omp single nowait
            {
                #pragma omp task
                merge_sort(arr, mid + 1, right);
            }
        }

        // Merge the sorted halves
        merge(arr, left, mid, right);
    }
}