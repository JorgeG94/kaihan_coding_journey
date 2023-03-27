#include <stdio.h>

#define N 1000

__global__ void matrix_multiply(float *a, float *b, float *c) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int index = i * N + j;
    if (i < N && j < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            sum += a[i*N+k] * b[k*N+j];
        }
        c[index] = sum;
    }
}

int main() {
    float *a, *b, *c;
    float *d_a, *d_b, *d_c;
    int size = N * N * sizeof(float);

    // Allocate memory on the host
    a = (float*)malloc(size);
    b = (float*)malloc(size);
    c = (float*)malloc(size);
    // can also be done using cudaMallocHost 

    // Initialize matrices a and b
    for (int i = 0; i < N*N; i++) {
        a[i] = 1.0f;
        b[i] = 2.0f;
    }

    // Allocate memory on the device
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    // Copy matrices a and b from host to device
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    // Define block and grid sizes
    dim3 blockDim(32, 32);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (N + blockDim.y - 1) / blockDim.y);

    // Launch kernel to multiply matrices a and b
    matrix_multiply<<<gridDim, blockDim>>>(d_a, d_b, d_c);

    // Copy result matrix c from device to host
    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    // Verify the result
    for (int i = 0; i < N*N; i++) {
        if (c[i] != 2.0f * N) {
            printf("Error: mismatch at index %d, expected %f, but got %f\n", i, 2.0f * N, c[i]);
            break;
        }
    }

    // Free memory on the device
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    // Free memory on the host
    free(a);
    free(b);
    free(c);

    return 0;
}

