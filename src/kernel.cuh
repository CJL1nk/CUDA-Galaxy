#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include "types.h"

__global__ void skibidiKernel(const int* a, const int* b, int* c, int n);

__global__ void clearDensityMap(int* densityMap, int numCells);

__global__ void gridBodies(const float* bodyXPos, const float* bodyYPos, int* bodyCells, int numBodies, float distancePerPixel, int screenWidth);

__global__ void generateDensityMap(const int* bodyCells, int* densityMap, int numBodies);

__global__ void generateHeatMap(const int* densityMap, uint8_t* heatMap, int numCells);
