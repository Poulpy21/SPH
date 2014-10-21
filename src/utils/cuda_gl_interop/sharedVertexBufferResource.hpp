

#ifndef CUDAGLSHAREDVERTEXBUFFERRESOURCE_H
#define CUDAGLSHAREDVERTEXBUFFERRESOURCE_H

#include <cassert>
#include "sharedBufferResource.hpp"

namespace cuda_gl_interop {

    class SharedVertexBufferResource : public SharedBufferResource {

        public:
            explicit SharedVertexBufferResource(
                    size_t nBytes,
                    unsigned int flags = cudaGraphicsRegisterFlagsNone,
                    GLenum usage = GL_STREAM_DRAW) :
                SharedBufferResource(GL_ARRAY_BUFFER, nBytes, flags, usage) 
        {
        }

            virtual ~SharedVertexBufferResource() {}
    };
}

#endif /* end of include guard: CUDAGLSHAREDVERTEXBUFFERRESOURCE_H */
