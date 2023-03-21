#include <iostream> 
#include <omp.h>

int main(){

int x[12];

for(int i = 0; i < 12; ++i){
  x[i] = i;
}

int r = 0;
#pragma omp parallel 
{
int tid = omp_get_thread_num();
if(tid == 0)
std::cout << " num threads is " << omp_get_num_threads() << std::endl;
#pragma omp for reduction(+:r)
for(int i = 0; i < 12; ++i){
 r += x[i] * x[i];
}
}
std::cout << " r is " << r << std::endl;

return 1;
}
