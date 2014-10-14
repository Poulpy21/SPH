
#include "implicitFunc.h"
#include "cudaUtils.hpp"

__device__ __host__ float d1(float x, float y, float z) { return x; }
__device__ __host__ float d2(float x, float y, float z) { return y; }
__device__ __host__ float d3(float x, float y, float z) { return z; }

__device__ __host__ float op1(float d1, float d2) { return d1+d2; } 
__device__ __host__ float op2(float d1, float d2) { return d1-d2; } 

const unsigned int nDensityFunctions = 3u;
const unsigned int nOperatorFunctions = 2u;

const densityFunction density_functions_p_h[nDensityFunctions] = {d1,d2,d3};
__device__ const densityFunction density_functions_p_d[nDensityFunctions] = {d1,d2,d3};

const operatorFunction operator_functions_p_h[nOperatorFunctions] = {op1,op2};
__device__ const operatorFunction operator_functions_p_d[nOperatorFunctions] = {op1,op2};

std::map<operatorFunction, operatorFunction> operatorFunctionPointers;
std::map<densityFunction, densityFunction> densityFunctionPointers;

__host__ void initPointers() {

    //densities
    densityFunction density_functions_p_d_h[nDensityFunctions];
    CHECK_CUDA_ERRORS(cudaMemcpyFromSymbol(
                density_functions_p_d_h, 
                density_functions_p_d, 
                nDensityFunctions*sizeof(densityFunction),
                0,cudaMemcpyDeviceToHost));
    
    for (unsigned int i = 0u; i < nDensityFunctions; i++) {
        densityFunctionPointers.insert(
                std::pair<densityFunction,densityFunction>(density_functions_p_h[i], density_functions_p_d_h[i])
                );        
    }
   
    //operators
    operatorFunction operator_functions_p_d_h[nOperatorFunctions];
    CHECK_CUDA_ERRORS(cudaMemcpyFromSymbol(
                operator_functions_p_d_h, 
                operator_functions_p_d, 
                nOperatorFunctions*sizeof(operatorFunction),
                0,cudaMemcpyDeviceToHost));
    
    for (unsigned int i = 0u; i < nOperatorFunctions; i++) {
        operatorFunctionPointers.insert(
                std::pair<operatorFunction,operatorFunction>(operator_functions_p_h[i], operator_functions_p_d_h[i])
                );        
    }
}
