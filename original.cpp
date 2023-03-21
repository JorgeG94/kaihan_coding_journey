#include <iostream>
#include <vector>
#include <chrono>
#include <cstdlib>


using namespace std;

int main() {
    // Define the two random matrices
    const int ROWS = 2;
    const int COLS = 2;

    // Seed the random number generator
    srand(time(0));

    int matrix1[ROWS][COLS];
    int matrix2[ROWS][COLS];

    // Initialize the matrices with random values from 0 to 9
    for (int i = 0; i < ROWS; i++) {
        for (int j = 0; j < COLS; j++) {
            matrix1[i][j] = rand() % 10; // generates a random number between 0 and 9
            matrix2[i][j] = rand() % 10;
        }
    }
   
    int result[ROWS][COLS];


    // Set the number of threads to be used in the parallel region
    //int num_threads = omp_get_num_procs();
    //omp_set_num_threads(num_threads);


    //#pragma omp parallel for
    for (int i = 0; i < ROWS; i++) {
        for (int j = 0; j < COLS; j++) {
            result[i][j] = 0;
            for (int k = 0; k < COLS; k++) {
                result[i][j] += matrix1[i][k] * matrix2[k][j];
            }
        }
    }

    for (int i = 0; i < ROWS; i++) {
        for (int j = 0; j < COLS; j++) {
            std::cout << matrix1[i][j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << " xxx " << std::endl;
    for (int i = 0; i < ROWS; i++) {
        for (int j = 0; j < COLS; j++) {
            std::cout << matrix2[i][j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << " xxx " << std::endl;
    // Print the execution time
    // Print the execution time

    for (int i = 0; i < ROWS; i++) {
        for (int j = 0; j < COLS; j++) {
            std::cout << result[i][j] << " ";
        }
        std::cout << std::endl;
    }
    // Print the execution time

    return 0;
}
