#pragma once

#include <cuda_runtime.h>

__global__ void skibidiKernel(const int* a, const int* b, int* c, int n);