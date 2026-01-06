#include "kernel.cuh"

__global__ void skibidiKernel(const int* a, const int* b, int* c, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
        c[index] = a[index] + b[index];
    }
}

__global__ void clearDensityMap(int* densityMap, const int numCells) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= numCells) return; // Exit early if body index is out of bounds

    densityMap[idx] = 0;
}

__global__ void gridBodies(const float* bodyXPos, const float* bodyYPos, int* bodyCells, const int numBodies,  const float distancePerPixel, const int screenWidth) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= numBodies) return; // Exit early if body index is out of bounds

    const int screenX = static_cast<int>(bodyXPos[idx] / distancePerPixel);
    const int screenY = static_cast<int>(bodyYPos[idx] / distancePerPixel);

    bodyCells[idx] = screenY * screenWidth + screenX;
}

__global__ void generateDensityMap(const int* bodyCells, int* densityMap, const int numBodies) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= numBodies) return; // Exit early if body index is out of bounds

    atomicAdd(&densityMap[bodyCells[idx]], 1);
}

__global__ void generateHeatMap(const int* densityMap, uint8_t* heatMap, const int numCells) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= numCells) return; // Exit early if body index is out of bounds

    float normalized = densityMap[idx] / 10.0f;
    if (normalized > 1.0f) normalized = 1.0f;
    if (normalized < 0.0f) normalized = 0.0f;

    // * 4 to account for 4 bytes
    const int pixelStart = idx * 4;
    heatMap[pixelStart + 2] = static_cast<uint8_t>(255.f * normalized);
    heatMap[pixelStart + 3] = static_cast<uint8_t>(255);
}

