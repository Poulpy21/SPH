
#ifndef IMPLICITFUNC_H
#define IMPLICITFUNC_H

#ifdef __CUDACC__
#define __HOST__ __host__
#define __DEVICE__ __device__
#else
#define __HOST__
#define __DEVICE__
#endif /* end if __CUDACC__ */

#include <map>
#include <cuda.h>
#include <cuda_runtime.h>

typedef float(*operatorFunction)(float,float);
typedef float(*densityFunction)(float,float,float);

//Densities
__HOST__ __DEVICE__ float d1(float x, float y, float z);
__HOST__ __DEVICE__ float d2(float x, float y, float z);
__HOST__ __DEVICE__ float d3(float x, float y, float z);

//Operators
__HOST__ __DEVICE__ float op1(float d1, float d2);
__HOST__ __DEVICE__ float op2(float d1, float d2);

//Pointers
extern std::map<operatorFunction, operatorFunction> operatorFunctionPointers;
extern std::map<densityFunction, densityFunction> densityFunctionPointers;

__HOST__ void initPointers();

#undef __HOST__
#undef __DEVICE__

#endif /* end of include guard: IMPLICITFUNC_H */
