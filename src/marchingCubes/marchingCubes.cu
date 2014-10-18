
#ifdef __CUDACC__

#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include "cudaUtils.hpp"
#include "defines.hpp"

namespace MarchingCubes {


    __global__ void 
    __launch_bounds__(512)
    computeDensitiesKernel(float x0, float y0, float z0,
            unsigned int W, unsigned int H, unsigned int L, 
            float h,
            cudaSurfaceObject_t densitiesSurface) {
        
        unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned int iy = blockIdx.y * blockDim.y + threadIdx.y;
        unsigned int iz = blockIdx.z * blockDim.z + threadIdx.z;

        if(ix >= W || iy >= H || iz >= L)
            return;

        /*surf3Dwrite(__float2half_rn(0.5f), densitiesSurface, ix/2*sizeof(float), iy, iz, cudaBoundaryModeTrap);*/
        surf3Dwrite(0.5f, densitiesSurface, ix*sizeof(float), iy, iz, cudaBoundaryModeTrap);
    }

    __host__ void callComputeDensitiesKernel(
        float x0, float y0, float z0,
        unsigned int W, unsigned int H, unsigned int L, 
        float h, 
        cudaSurfaceObject_t densitiesSurface) {
       
        dim3 blockDim(8,8,8);
        dim3 gridDim(
                (W+blockDim.x-1)/blockDim.x, 
                (H+blockDim.y-1)/blockDim.y, 
                (L+blockDim.z-1)/blockDim.z);


        computeDensitiesKernel<<<gridDim,blockDim,0>>>(x0,y0,z0,W,H,L,h,densitiesSurface);
        CHECK_KERNEL_EXECUTION();
    }
}

#endif /* ifdef __CUDACC__ */

