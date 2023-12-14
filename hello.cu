#include <stdio.h>

__global__ void cuda_hello(){
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    printf("Hello World from thread %d, %d\n", x, y);
}

int main() {
    
    dim3 g(2, 5);
    cuda_hello<<<g,2>>>();
    
    cudaDeviceReset(); 
    return 0;
}
