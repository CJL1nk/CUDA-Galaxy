#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include "types.h"

__global__ void skibidiKernel(const int* a, const int* b, int* c, int n);

__global__ void clearBodyCells(int* bodyCells, int numBodies);

__global__ void clearDensityMap(int* densityMap, uint8_t* heatMap, int numCells);

__global__ void gridBodies(const float* bodyXPos, const float* bodyYPos, int* bodyCells, int numBodies, float distancePerPixel, int screenWidth, int screenHeight);

__global__ void generateDensityMap(const int* bodyCells, int* densityMap, int numBodies);

__global__ void physicsKernel(const float* bodyXPos, const float* bodyYPos, float* bodyXVel, float* bodyYVel, float* mass, int* bodyCells, int numBodies);

__global__ void positionKernel(float* bodyXPos, float* bodyYPos, const float* bodyXVel, const float* bodyYVel, int numBodies);

__global__ void generateHeatMap(const int* densityMap, uint8_t* heatMap, int numCells);
