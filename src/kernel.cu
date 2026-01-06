#include "kernel.cuh"

__global__ void skibidiKernel(const int* a, const int* b, int* c, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
        c[index] = a[index] + b[index];
    }
}

__global__ void clearDensityMap(int* densityMap, uint8_t* heatMap, const int numCells) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= numCells) return; // Exit early if body index is out of bounds

    densityMap[idx] = 0;
    heatMap[idx] = 0;
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

__global__ void physicsKernel(const float* bodyXPos, const float* bodyYPos, float* bodyXVel, float* bodyYVel, int* bodyCells, int numBodies, const int threadsPerBlock) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= numBodies) return; // Exit early if body index is out of bounds

    __shared__ float sx[512];
    __shared__ float sy[512];

    // Acceleration
    float ax = 0.0f;
    float ay = 0.0f;

    for (int tile = 0; tile < (numBodies + threadsPerBlock) / threadsPerBlock; tile++) {

        int j = tile * threadsPerBlock + threadIdx.x;

        if (j < numBodies) {
            sx[threadIdx.x] = bodyXPos[j];
            sy[threadIdx.x] = bodyYPos[j];
        }
        __syncthreads();

        int tileSize = min(threadsPerBlock, numBodies - tile * threadsPerBlock);

        for (int k = 0; k < tileSize; k++) {
            int bodyJ = tile * threadsPerBlock + k;
            if (bodyJ == idx) continue;

            float dx = sx[k] - bodyXPos[idx];
            float dy = sy[k] - bodyYPos[idx];
            float r2 = dx*dx + dy*dy + 1e-6f;
            float inv = rsqrtf(r2);

            ax += dx * inv;
            ay += dy * inv;
        }
        __syncthreads();
    }

    bodyXVel[idx] += ax;
    bodyYVel[idx] += ay;
}

__global__ void positionKernel(float* bodyXPos, float* bodyYPos, const float* bodyXVel, const float* bodyYVel, int numBodies) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= numBodies) return; // Exit early if body index is out of bounds

    bodyXPos[idx] += bodyXVel[idx];
    bodyYPos[idx] += bodyYVel[idx];
}

__global__ void generateHeatMap(const int* densityMap, uint8_t* heatMap, const int numCells) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= numCells) return; // Exit early if body index is out of bounds

    float normalized = densityMap[idx] / 25.0f;
    if (normalized > 1.0f) normalized = 1.0f;
    if (normalized < 0.0f) normalized = 0.0f;

    // * 4 to account for 4 bytes
    const int pixelStart = idx * 4;
    heatMap[pixelStart + 2] = static_cast<uint8_t>(255.f * normalized);
    heatMap[pixelStart + 3] = static_cast<uint8_t>(255);
}

