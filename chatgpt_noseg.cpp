#include <iostream>
#include <chrono>
#include <omp.h>
using namespace std;

const int SIZE = 4000;

void multiply_matrices(int** A, int** B, int** C) {
    int num_threads;
    #pragma omp parallel for collapse(3) num_threads(16) 
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            for (int k = 0; k < SIZE; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}



int main() {
    // allocate matrices A, B, and C dynamically
    int** A = new int*[SIZE];
    int** B = new int*[SIZE];
    int** C = new int*[SIZE];
    for (int i = 0; i < SIZE; i++) {
        A[i] = new int[SIZE];
        B[i] = new int[SIZE];
        C[i] = new int[SIZE];
    }

    // initialize matrices A and B
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            A[i][j] = i + j;
            B[i][j] = i - j;
	    C[i][j] = 0;
        }
    }

    // multiply matrices and measure execution time
    auto start = chrono::steady_clock::now();
    multiply_matrices(A, B, C);
    auto end = chrono::steady_clock::now();
    auto diff = end - start;

    // output execution time
    cout << "Matrix multiplication took "
         << chrono::duration <double, milli> (diff).count()
         << " ms." << endl;

    // deallocate matrices
    for (int i = 0; i < SIZE; i++) {
        delete[] A[i];
        delete[] B[i];
        delete[] C[i];
    }
    delete[] A;
    delete[] B;
    delete[] C;

    return 0;
}

