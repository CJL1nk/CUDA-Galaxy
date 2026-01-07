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
}

__global__ void gridBodies(const float* bodyXPos, const float* bodyYPos, int* bodyCells, const int numBodies,  const float distancePerPixel, const int screenWidth, const int screenHeight) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= numBodies) return; // Exit early if body index is out of bounds

    const int screenX = static_cast<int>(bodyXPos[idx] / distancePerPixel);
    const int screenY = static_cast<int>(bodyYPos[idx] / distancePerPixel);

    if (screenX < 0 || screenX >= screenWidth || screenY < 0 || screenY >= screenHeight) {
        bodyCells[idx] = -1;
        return;
    }

    bodyCells[idx] = screenY * screenWidth + screenX;
}

__global__ void generateDensityMap(const int* bodyCells, int* densityMap, const int numBodies) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= numBodies) return; // Exit early if body index is out of bounds

    int cell = bodyCells[idx];

    if (cell >= 0) {
        atomicAdd(&densityMap[cell], 1);
    }
}

__global__ void physicsKernel(const float* bodyXPos, const float* bodyYPos, float* bodyXVel, float* bodyYVel, float* mass, int* bodyCells, int numBodies) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= numBodies) return; // Exit early if body index is out of bounds

    constexpr float G  = 100.f;
    constexpr float dt = 0.005f;

    // Shared mem W speed
    extern __shared__ float shmem[];
    float* sx = shmem;
    float* sy = shmem + blockDim.x;
    float* sm = shmem + 2 * blockDim.x;

    // X Index
    const float xi = bodyXPos[idx];
    const float yi = bodyYPos[idx];

    // Acceleration
    float ax = 0.0f;
    float ay = 0.0f;

    int numTiles = (numBodies + blockDim.x - 1) / blockDim.x;

    for (int tile = 0; tile < numTiles; tile++) {

        int j = tile * blockDim.x + threadIdx.x;
        if (j < numBodies) {
            sx[threadIdx.x] = bodyXPos[j];
            sy[threadIdx.x] = bodyYPos[j];
            sm[threadIdx.x] = mass[j];
        } else {
            sx[threadIdx.x] = 0.0f;
            sy[threadIdx.x] = 0.0f;
            sm[threadIdx.x] = 0.0f;
        }
        __syncthreads();

        int tileSize = min(blockDim.x, numBodies - tile * blockDim.x);

        for (int k = 0; k < tileSize; k++) { // RIP performance

            int bodyJ = tile * blockDim.x + k;
            if (bodyJ == idx) continue;

            float dx = sx[k] - xi;
            float dy = sy[k] - yi;

            float r2 = dx * dx + dy * dy + 50.f; // Softening
            float invR = rsqrtf(r2);
            float invR3 = invR * invR * invR;

            float s = G * sm[k] * invR3;

            ax += dx * s;
            ay += dy * s;
        }
        __syncthreads();
    }

    bodyXVel[idx] += ax * dt;
    bodyYVel[idx] += ay * dt;
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

    float normalized = densityMap[idx] / 5.0f;
    if (normalized > 1.0f) normalized = 1.0f;
    if (normalized < 0.0f) normalized = 0.0f;

    float whiteAmount = 0.0f;

    if (normalized >= 0.85f) {
        whiteAmount = static_cast<uint8_t>(255 - (normalized * 128));
    }

    // * 4 to account for 4 bytes
    const int pixelStart = idx * 4;
    heatMap[pixelStart + 0] = static_cast<uint8_t>(whiteAmount);
    heatMap[pixelStart + 1] = static_cast<uint8_t>(127.f * normalized + whiteAmount);
    heatMap[pixelStart + 2] = static_cast<uint8_t>(255.f * normalized);
    heatMap[pixelStart + 3] = static_cast<uint8_t>(255);
}

