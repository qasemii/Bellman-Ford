// assert num_edge are different

#include <stdio.h>


void save_report() {
    FILE* report_file = fopen("report.txt", "a");

    if (report_file == NULL) {
        // File doesn't exist, create and write header
        report_file = fopen("report.txt", "w");
    }
    fclose(report_file);
}

int main() {
    FILE *file1, *file2;
    char line1[100], line2[100];
    int lineNumber = 0;
    int mismatch = 0;

    // Open the first file
    file1 = fopen("cuda_output.txt", "r");
    if (!file1) {
        perror("Error opening cuda_output.txt");
        return 1;
    }

    // Open the second file
    file2 = fopen("omp_output.txt", "r");
    if (!file2) {
        perror("Error opening file2.txt");
        fclose(file1);
        return 1;
    }

    printf("CUDA vs OpenMP------------------\n");
    // Compare each line in the files
    while (fgets(line1, sizeof(line1), file1) && fgets(line2, sizeof(line2), file2)) {
        int num1, num2;

        // Convert strings to integers
        sscanf(line1, "%d", &num1);
        sscanf(line2, "%d", &num2);

        // Compare the numbers
        if (num1 != num2) {
            printf("Mismatch  at line\t%d\n", lineNumber);
            mismatch++;
        }
        lineNumber++;
    }
    printf("Total mismatch answers:\t%d\n\n", mismatch);
    printf("--------------------------------\n");

    fclose(file1);
    fclose(file2);

    return 0;
}
