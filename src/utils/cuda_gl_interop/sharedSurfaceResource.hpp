

#ifndef CUDAGLSHAREDSURFACERESOURCE_H
#define CUDAGLSHAREDSURFACERESOURCE_H

#include <cassert>
#include "sharedImageResource.hpp"

namespace cuda_gl_interop {

    class SharedSurfaceResource : public SharedImageResource {

        public:
            explicit SharedSurfaceResource(
                    GLenum target, 
                    unsigned int nLevels,
                    unsigned int internalFormat, 
                    unsigned int width, unsigned int height, unsigned int depth) :
                
                SharedImageResource(target, nLevels, internalFormat, 
                        width, height, depth, 
                        cudaGraphicsRegisterFlagsSurfaceLoadStore)
            {
            }
            
            virtual ~SharedSurfaceResource() {}

            cudaSurfaceObject_t createSurfaceObject(unsigned int arrayIndex, unsigned int mipLevel) {
                
                cudaSurfaceObject_t surface = 0;
                cudaArray_t array = this->getCudaArray(arrayIndex, mipLevel);
        
                //Cuda ressource descriptor
                cudaResourceDesc resDesc;
                memset(&resDesc, 0, sizeof(cudaResourceDesc));
                resDesc.resType = cudaResourceTypeArray; 
                resDesc.res.array.array = array;

                CHECK_CUDA_ERRORS(cudaCreateSurfaceObject(&surface, &resDesc));

                return surface;
            }

            
    };
}

#endif /* end of include guard: CUDAGLSHAREDSURFACERESOURCE_H */
