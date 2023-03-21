#include <iostream>
#include <omp.h>
#include <vector>
#include <chrono>
#include <cstdlib>
#include <cassert>



int main() {
    // Define the two random matrices
    const int ROWS = 2000;
    const int COLS = 2000;

    auto matrix1 = new int [ROWS][COLS];
    auto matrix2 = new int [ROWS][COLS];
    auto result_serial = new int [ROWS][COLS];
    auto result = new int [ROWS][COLS];
    
    double start2 = omp_get_wtime();
    #pragma omp parallel for collapse(2) num_threads(10)
    for(int i = 0; i < ROWS; ++i){
      for(int j = 0; j < COLS; ++j){
        matrix1[i][j] = 0;
        matrix2[i][j] = 0;
        result[i][j] = 0;
        result_serial[i][j] = 0;
      }
    }

    // Seed the random number generator
    srand(time(0));

    // Initialize the matrices with random values from 0 to 9
    for (int i = 0; i < ROWS; i++) {
        for (int j = 0; j < COLS; j++) {
            matrix1[i][j] = rand() %10; // generates a random number between 0 and 9
            matrix2[i][j] = rand() % 10;
        }
    }
    printf("time for init: %f s\n", (omp_get_wtime() - start2));
  	


    // Set the number of threads to be used in the parallel region

	double start1 = omp_get_wtime();
    #pragma omp parallel for collapse(3) num_threads(1)
    for (int i = 0; i < ROWS; i++) {
        for (int j = 0; j < COLS; j++) {
            for (int k = 0; k < COLS; k++) {
                result_serial[i][j] += matrix1[i][k] * matrix2[k][j];
            }
        }
    }
    printf("time for serial: %f s\n", (omp_get_wtime() - start1));


	double start = omp_get_wtime();
    #pragma omp parallel for collapse(3) num_threads(12)
    for (int i = 0; i < ROWS; i++) {
        for (int j = 0; j < COLS; j++) {
            for (int k = 0; k < COLS; k++) {
                result[i][j] += matrix1[i][k] * matrix2[k][j];
            }
        }
    }
    printf("time omp: %f s\n", (omp_get_wtime() - start));


    for (int i = 0; i < ROWS; i++) {
        for (int j = 0; j < COLS; j++) {
		assert(result_serial[i][j] == result[i][j]);
        }
    }

    delete [] matrix1;
    delete [] matrix2;
    delete [] result_serial;
    delete [] result;

    return 0;
}
