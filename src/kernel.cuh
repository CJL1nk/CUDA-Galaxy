#pragma once

#include <cuda_runtime.h>
#include "types.h"

__global__ void skibidiKernel(const int* a, const int* b, int* c, int n);
__global__ void updateBodyPositions(Body* bodies, size_t bodiesSize, int* densityMap, size_t mapSize);