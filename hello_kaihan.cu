#include <stdio.h>

__global__ void print_hello() {
    printf("Hello Kaihan\n");
}

int main() {
    print_hello<<<1,1>>>();
    cudaDeviceSynchronize();
    return 0;
}
