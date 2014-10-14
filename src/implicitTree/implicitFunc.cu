#include "implicitFunc.h"

__device__ __host__ float d1(float x, float y, float z) { return x; }
__device__ __host__ float d2(float x, float y, float z) { return y; }
__device__ __host__ float d3(float x, float y, float z) { return z; }

__device__ __host__ float op1(float d1, float d2) { return d1+d2; } 
__device__ __host__ float op2(float d1, float d2) { return d1-d2; } 

