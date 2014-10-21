

#ifndef CUDAGLSHAREDPIXELBUFFERRESOURCE_H
#define CUDAGLSHAREDPIXELBUFFERRESOURCE_H

#include <cassert>
#include "sharedBufferResource.hpp"

namespace cuda_gl_interop {

    class SharedPixelBufferResource : public SharedBufferResource {

        public:
            explicit SharedPixelBufferResource(
                    size_t nBytes,
                    bool isPacked,
                    unsigned int flags = cudaGraphicsRegisterFlagsNone,
                    GLenum usage = GL_STREAM_DRAW) :
                SharedBufferResource(isPacked ? GL_PIXEL_PACK_BUFFER : GL_PIXEL_UNPACK_BUFFER, nBytes, flags, usage),
                _isPacked(isPacked)
        {
        }

            virtual ~SharedPixelBufferResource() {}

            bool isPacked() {
                return _isPacked;
            }

        private:
            bool _isPacked;
    };
}

#endif /* end of include guard: CUDAGLSHAREDPIXELBUFFERRESOURCE_H */
