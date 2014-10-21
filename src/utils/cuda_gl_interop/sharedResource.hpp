
#ifndef CUDAGLSHAREDRESOURCE_H
#define CUDAGLSHAREDRESOURCE_H

#include <cassert>
#include "headers.hpp"
#include "log.hpp"

namespace cuda_gl_interop {

    class SharedResource {
        public:
            virtual ~SharedResource() {
                if(_isMapped)
                    unmapResource();
                if(_isRegistered)
                    unregisterResource();
            };

            void mapResource(cudaStream_t stream = 0) {
                assert(_isRegistered);
                if(_isMapped) {
                    log_console->warnStream() << "[OpenGLCUDASharedResource] Resource was already mapped !";
                }
                else {
                    CHECK_CUDA_ERRORS(cudaGraphicsMapResources(1, &_cudaResource, stream));
                    _isMapped = true;
                }
            }

            void unmapResource(cudaStream_t stream = 0) {
                assert(_isRegistered);
                if(_isMapped) {
                    CHECK_CUDA_ERRORS(cudaGraphicsUnmapResources(1, &_cudaResource, stream));
                    _isMapped = false;
                }
                else {
                    log_console->warnStream() << "[OpenGLCUDASharedResource] Resource was already unmapped !";
                }
            }
            
            bool isRegistered() const {
                return _isRegistered;
            }

            bool isMapped() const {
                return _isMapped;
            }

            unsigned int getGlResource() const {
                return _glResource;
            }

            cudaGraphicsResource_t getCudaResource() const {
                return _cudaResource;
            }
        
        protected:
            SharedResource() :
                _glResource(0),
                _cudaResource(0),
                _isRegistered(false),
                _isMapped(false) {
            }
            
            void registerResource() {
                assert(!_isMapped);
                if(_isRegistered) {
                    log_console->warnStream() << "[OpenGLCUDASharedResource] Resource was already registered !";
                }
                else {
                    registerInternalResource();
                    _isRegistered = true;
                }
            }

            void unregisterResource() {
                assert(!_isMapped);
                if(_isRegistered) {
                    CHECK_CUDA_ERRORS(cudaGraphicsUnregisterResource(_cudaResource));
                    _isRegistered = false;
                }
                else {
                    log_console->warnStream() << "[OpenGLCUDASharedResource] Resource was already unregistered !";
                }
            }
            
    
            virtual void registerInternalResource() = 0;

        protected:
            unsigned int _glResource;
            cudaGraphicsResource_t _cudaResource;
        private:
            bool _isRegistered;
            bool _isMapped;
    };
}

#endif /* end of include guard: CUDAGLSHAREDRESOURCE_H */
