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

__global__ void generateHeatMap(const int* densityMap, uint8_t* heatMap, float* d_prevNormalized, const int numCells) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= numCells) return; // Exit early if body index is out of bounds

    float raw = densityMap[idx] / 20.0f;
    raw = fminf(fmaxf(raw, 0.0f), 1.0f);

    // Smooth only brightness
    float prevV = d_prevNormalized[idx];
    float vSmooth = lerp1(prevV, raw, 0.15f);
    d_prevNormalized[idx] = vSmooth;;

    float h, s, v;

    if (raw < 0.8f) {
        float t = raw / 0.8f;
        h = (1.0f - t) * 240.0f;
        s = 1.0f;
        v = sqrtf(vSmooth);
    } else {
        float t = (raw - 0.8f) / 0.2f;
        h = 0.0f;
        s = 1.0f - t;
        v = 1.0f;
    }

    float r, g, b;
    hsvToRgb(h, s, v, r, g, b);

    int pixelStart = idx * 4;
    heatMap[pixelStart + 0] = (uint8_t)(r * 255.0f);
    heatMap[pixelStart + 1] = (uint8_t)(g * 255.0f);
    heatMap[pixelStart + 2] = (uint8_t)(b * 255.0f);
    heatMap[pixelStart + 3] = 255;
}

__device__ void hsvToRgb(float h, float s, float v, float& r, float& g, float& b) {

    h = fmodf(h, 360.0f);
    if (h < 0.0f) h += 360.0f;

    float c = v * s;
    float hPrime = h / 60.0f;
    float x = c * (1.0f - fabsf(fmodf(hPrime, 2.0f) - 1.0f));

    float r1, g1, b1;

    if      (hPrime < 1.0f) { r1 = c; g1 = x; b1 = 0.0f; }
    else if (hPrime < 2.0f) { r1 = x; g1 = c; b1 = 0.0f; }
    else if (hPrime < 3.0f) { r1 = 0.0f; g1 = c; b1 = x; }
    else if (hPrime < 4.0f) { r1 = 0.0f; g1 = x; b1 = c; }
    else if (hPrime < 5.0f) { r1 = x; g1 = 0.0f; b1 = c; }
    else                    { r1 = c; g1 = 0.0f; b1 = x; }

    float m = v - c;
    r = r1 + m;
    g = g1 + m;
    b = b1 + m;
}

__device__ __forceinline__ float lerp1(float a, float b, float t) {
    return a + t * (b - a);
}

