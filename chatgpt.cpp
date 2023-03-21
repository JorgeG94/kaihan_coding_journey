#include <iostream>
#include <chrono>

using namespace std;

const int SIZE = 1000;

void multiply_matrices(int A[][SIZE], int B[][SIZE], int C[][SIZE]) {
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            C[i][j] = 0;
            for (int k = 0; k < SIZE; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

int main() {
    int A[SIZE][SIZE];
    int B[SIZE][SIZE];
    int C[SIZE][SIZE];

    // initialize matrices A and B
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            A[i][j] = i + j;
            B[i][j] = i - j;
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

    return 0;
}

