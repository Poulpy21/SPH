


#ifndef CUDAGLSHAREDBUFFERRESOURCE_H
#define CUDAGLSHAREDBUFFERRESOURCE_H

#include <cassert>
#include "sharedResource.hpp"

namespace cuda_gl_interop {

    class SharedBufferResource : public SharedResource {

        public:
            virtual ~SharedBufferResource() {
                glDeleteBuffers(1, &(this->_glResource));
                _device_ptr = 0;
            };
            
            unsigned int getBufferID() {
                return this->getGlResource();
            }

            size_t getSize() {
                return _nBytes;
            }
            
            GLenum getTarget() {
                return _target;
            }

            void* getDevicePointer() {
                assert(this->isMapped());
                return _device_ptr;
            }

        protected:
            explicit SharedBufferResource(
                    GLenum target, 
                    size_t nBytes,
                    unsigned int flags = cudaGraphicsRegisterFlagsNone,
                    GLenum usage = GL_STREAM_DRAW) :
                SharedResource(),
                _target(target),
                _nBytes(nBytes),
                _flags(flags),
                _device_ptr(0)
        {
            assert(    target == GL_ARRAY_BUFFER
                    || target == GL_PIXEL_UNPACK_BUFFER
                    || target == GL_PIXEL_PACK_BUFFER);

            glGenBuffers(1, &(this->_glResource));
            glBindBuffer(_target, this->getGlResource());
            glBufferData(_target, _nBytes, NULL, usage);
            glBindBuffer(_target, 0);

            assert(glIsBuffer(this->getGlResource()));

            this->registerResource();

            this->mapResource();
            {
                size_t size = 0;
                cudaGraphicsResourceGetMappedPointer(&_device_ptr, &size, this->_cudaResource);
                assert(size == _nBytes);
            }
            this->unmapResource();

            assert(_device_ptr != 0);
        }


        private:
            GLenum _target;
            size_t _nBytes;
            unsigned int _flags;
            void* _device_ptr;
            
            void registerInternalResource() {
                CHECK_CUDA_ERRORS(cudaGraphicsGLRegisterBuffer(&(this->_cudaResource), this->getGlResource(), _flags));
            }
    };
}

#endif /* end of include guard: CUDAGLSHAREDBUFFERRESOURCE_H */
