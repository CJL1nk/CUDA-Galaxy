#include "kernel.cuh"

__global__ void skibidiKernel(const int* a, const int* b, int* c, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
        c[index] = a[index] + b[index];
    }
}

__global__ void updateBodyPositions(Body* bodies, size_t bodiesSize, int* densityMap, size_t mapSize) {

}