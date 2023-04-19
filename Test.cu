#include <iostream>
#include <omp.h>
#include <vector>
#include <chrono>
#include <cstdlib>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <time.h>


__global__ void multiply_matrix_gpu(double* A, double* B, double* C, int size) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < size && col < size) {
        double sum = 0;
        for (int i = 0; i < size; ++i) {
            sum += A[row * size + i] * B[i * size + col];
        }
        C[row * size + col] = sum;
    }
}

using namespace std;
using namespace std::chrono;

int main() {
    const int N = 1000; // Size of the matrix
    double *A, *B, *C;
    double *dev_A, *dev_B, *dev_C;

// Allocate memory for the matrices
    A = new double[N * N];
    B = new double[N * N];
    C = new double[N * N];

// Fill the matrices with random values
    srand(time(NULL));
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            A[i * N + j] = (double)rand() / RAND_MAX * 9;
            // Generate a random value between 0 and 9
            B[i * N + j] = (double)rand() / RAND_MAX * 9;
        }
    }

// Print matrix A
    std::cout << "Matrix A:\n";
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            std::cout << A[i * N + j] << " ";
        }
        std::cout << std::endl;
    }
// Print matrix B
    std::cout << "Matrix B:\n";
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            std::cout << B[i * N + j] << " ";
        }
        std::cout << std::endl;
    }

// Allocate device memory for the matrices
    cudaMalloc((void**)&dev_A, N * N * sizeof(double));
    cudaMalloc((void**)&dev_B, N * N * sizeof(double));
    cudaMalloc((void**)&dev_C, N * N * sizeof(double));

// Copy the host matrices to the device
    cudaMemcpy(dev_A, A, N * N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_B, B, N * N * sizeof(double), cudaMemcpyHostToDevice);

// Set the device matrix C to all zeros using cudaMemset
    cudaMemset(dev_C, 0, N * N * sizeof(double));

// Define the block and grid sizes
    int block_size = 12;
    dim3 dim_block(block_size, block_size);
    dim3 dim_grid((N + dim_block.x - 1) / dim_block.x, (N + dim_block.y - 1) / dim_block.y);


// Start the timer
    auto start = chrono::steady_clock::now();

// Call the matrix multiplication kernel
    multiply_matrix_gpu<<<dim_grid, dim_block>>>(dev_A, dev_B, dev_C, N);

// Stop the timer
    auto end = chrono::steady_clock::now();
    auto diff = end - start;
// Copy the result matrix from the device to the host
    cudaMemcpy(C, dev_C, N * N * sizeof(double), cudaMemcpyDeviceToHost);

// Verify the result
 bool pass = true;
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            double sum = 0;
            for (int k = 0; k < N; ++k) {
                sum += A[i * N + k] * B[k * N + j];
            }
            if (abs(sum - C[i * N + j]) > 1e-6) { // Compare with a tolerance of 1e-6
                pass = false;
                break;
            }
        }
    }

    if (pass) {
        std::cout << "Verification passed!\n";
    } else {
        std::cout << "Verification failed!\n";
    }
// Print the resulting matrix
    std::cout << "Result matrix:\n";
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            std::cout << C[i * N + j] << " ";
        }
        std::cout << std::endl;
    }
// Print out the execution time
    cout << "Matrix multiplication took "
         << chrono::duration <double, milli> (diff).count()
         << " ms." << endl;

// Free device memory
    cudaFree(dev_A);
    cudaFree(dev_B);
    cudaFree(dev_C);

// Free host memory
    free(A);
    free(B);
    free(C);

    return 0;
}



