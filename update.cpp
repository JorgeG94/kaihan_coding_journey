#include <iostream>
#include <omp.h>
#include <vector>
#include <chrono>
#include <cstdlib>
using namespace std;
using namespace std::chrono;
int main() {
    // Define the two random matrices
    const int ROWS = 1000;
    const int COLS = 1000;
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
    //at this instant use function now()
    auto start = high_resolution_clock::now();
    
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
  //          std::cout << matrix1[i][j] << " ";
        }
    //    std::cout << std::endl;
    }
    std::cout << " xxx " << std::endl;
    for (int i = 0; i < ROWS; i++) {
        for (int j = 0; j < COLS; j++) {
//            std::cout << matrix2[i][j] << " ";
        }
  //      std::cout << std::endl;
    }
    std::cout << " xxx " << std::endl;
    for (int i = 0; i < ROWS; i++) {
        for (int j = 0; j < COLS; j++) {
    //        std::cout << result[i][j] << " ";
        }
    //    std::cout << std::endl;
    }
    
    // After function call
    auto stop = high_resolution_clock::now();       
    // Print the execution time of Parralel
    auto duration = duration_cast<microseconds>(stop - start);
    cout << duration.count() << endl;
    return 0;
}
