#include <iostream>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <omp.h>
#include <iomanip>
using namespace std;
const int n = 20000; // matrix size
 void multiply_matrices(int* A, int* B, int* C) {
        int num_threads;
        int thread_id;
        #pragma omp parallel 
	{
            thread_id = omp_get_thread_num();
            if(thread_id == 0) 
            std::cout << " we are using " << omp_get_num_threads() << " threads " << std::endl;
            
            #pragma omp for collapse(3) 
            for(int i = 0; i < n; i++) {
                for(int j = 0; j < n; j++) {
                    for(int k = 0; k < n; k++) {
                        C[i*n + j] += A[i*n + k] * B[k*n + j];
                    }
                }
            }
        }
    }  
    
int main(){
    // Allocated matrix A, B, C
    int *A = new int[n*n]; // first matrix
    int *B = new int[n*n]; // second matrix
    int *C = new int[n*n]; // result matrix
    // initialize the matrices with random values
    srand(time(NULL));
    for(int i = 0; i < n*n; i++) {
        A[i] = i;
        B[i] = i;
        C[i] = 0;
    }
   
   // multiply matrices and measure execution time
    
    // parallel region for matrix multiplication
    // double start1 = omp_get_wtime();
    auto start = chrono::steady_clock::now();
    // #pragma omp parallel for collapse(3) num_threads(1)
   
   multiply_matrices(A,B,C);

    // print the result matrix
    
    // multiply matrices and measure execution time
    
    auto end = chrono::steady_clock::now();
    auto diff = end - start;
    for(int i = 0; i < n; ++i){
    	C[i*n +i] += i;
    }
    // output execution time
    cout << "Matrix multiplication took "
         << std::setprecision(8) << chrono::duration <double, milli> (diff).count() * 0.001
         << " s" << endl;
    // deallocate memory for the matrices
    delete[] A;
    delete[] B;
    delete[] C;
    return 0;
}
