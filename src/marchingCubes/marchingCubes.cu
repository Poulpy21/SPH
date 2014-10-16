
#ifdef __CUDACC__

#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include "cudaUtils.hpp"
#include "defines.hpp"

namespace MarchingCubes {

    surface<void, cudaSurfaceType3D> densitiesSurface;
    surface<void, cudaSurfaceType3D> normalsSurface;

    __host__ void bindSurfaces(const cudaArray_t densitiesArray, const cudaArray_t normalsArray) {
        CHECK_CUDA_ERRORS(cudaBindSurfaceToArray(densitiesSurface, densitiesArray));
        CHECK_CUDA_ERRORS(cudaBindSurfaceToArray(normalsSurface, normalsArray));
    }
    __global__ void 
    __launch_bounds__(512)
    computeDensitiesKernel(float x0, float y0, float z0,
            unsigned int W, unsigned int H, unsigned int L, 
            float h) {
        
        unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned int iy = blockIdx.y * blockDim.y + threadIdx.y;
        unsigned int iz = blockIdx.z * blockDim.z + threadIdx.z;

        if(ix >= W || iy >= H || iz >= L)
            return;

        surf3Dwrite(h, densitiesSurface, ix*sizeof(float), iy, iz);
    }

    __host__ void callComputeDensitiesKernel(
            float x0, float y0, float z0,
            unsigned int W, unsigned int H, unsigned int L, 
            float h) {
       
        dim3 blockDim(8,8,8);
        dim3 gridDim(
                (W+blockDim.x-1)/blockDim.x, 
                (H+blockDim.y-1)/blockDim.y, 
                (L+blockDim.z-1)/blockDim.z);


        printf("ProbDim : (%i,%i,%i)\n", W, H, L); 
        printf("BlockDim : (%i,%i,%i)\n", blockDim.x, blockDim.y, blockDim.z); 
        printf("GridDim : (%i,%i,%i)\n", gridDim.x, gridDim.y, gridDim.z); 
        computeDensitiesKernel<<<gridDim,blockDim>>>(x0,y0,z0,W,H,L,h);
        CHECK_KERNEL_EXECUTION();
    }
}

#endif /* ifdef __CUDACC__ */

