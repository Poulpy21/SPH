

#include <cassert>
#include "headers.hpp"
#include "glUtils.hpp"
#include "defines.hpp"
#include "sharedResource.hpp"

#ifndef CUDAGLSHAREDIMAGERESOURCE_H
#define CUDAGLSHAREDIMAGERESOURCE_H

namespace cuda_gl_interop {

    class SharedImageResource : public SharedResource {

        public:
            virtual ~SharedImageResource() {};

        protected:
            explicit SharedImageResource(
                    GLenum target, 
                    unsigned int nLevels,
                    unsigned int internalFormat,
                    unsigned int width, unsigned int height, unsigned int depth,
                    unsigned int flags) :
                SharedResource(),
                _target(target),
                _nLevels(nLevels),
                _internalFormat(internalFormat),
                _width(width), _height(height), _depth(depth),
                _flags(flags)
        {
            assert(nLevels >= 1u);
            checkCudaCapabilities();

            //precompute some constants
            const GLenum validExternalFormat = utils::internalFormatToValidExternalFormat(internalFormat);
            const GLenum validExternalType = utils::internalFormatToValidExternalType(internalFormat);
            const unsigned int nChannels = utils::externalFormatToChannelNumber(validExternalFormat);
            const size_t nBytesPerChannel = utils::externalTypeToBytes(validExternalType);
            const size_t nBytesPerTexel = nChannels*nBytesPerChannel;
            const size_t nTexels = std::max(_width,1u) * std::max(_height,1u) * std::max(_depth,1u);

            log_console->infoStream() << "[SharedImageResource] Creating image resource...";
            log_console->infoStream() << "[SharedImageResource] \tTop level image size : " << utils::toStringVec3(_width, _height, _depth);
            log_console->infoStream() << "[SharedImageResource] \tInternal data format : " << utils::toStringInternalFormat(_internalFormat);
            log_console->debugStream() << "[SharedImageResource] \tExternal data format chosen : " << utils::toStringExternalFormat(validExternalFormat);
            log_console->debugStream() << "[SharedImageResource] \tExternal data type chosen : " << utils::toStringExternalType(validExternalType);
            log_console->debugStream() << "[SharedImageResource] \tNumber of channels : " << nChannels;
            log_console->debugStream() << "[SharedImageResource] \tNumber of bytes per channel : " << nBytesPerChannel;
            log_console->debugStream() << "[SharedImageResource] \tNumber of bytes per texel : " << nBytesPerTexel;
            log_console->debugStream() << "[SharedImageResource] \tNumber of top level texels : " << nTexels;

            //allocate initial 0's data
            void *junkData = calloc(nTexels, nBytesPerTexel);

            //create openGL texture and bind it to _TARGET
            glActiveTexture(GL_TEXTURE0);
            glGenTextures(1, &this->_glResource);
            glBindTexture(_target, this->getGlResource());

            //target dependant code
            unsigned long long int totalBytes = 0;
            unsigned int levelWidth, levelHeight, levelDepth;
            unsigned long long int levelBytes;
            switch(_target) {

                //1D TEXTURES
                case(GL_TEXTURE_1D):
                    assert(_width >= 1 && _height == 1 && _depth == 1);
                    glTexStorage1D(_target, _nLevels, _internalFormat, _width);
                    for (unsigned int i = 0; i < _nLevels; i++) {
                        levelWidth = std::max(1u, _width>>i);
                        levelHeight = 1u;
                        levelDepth = 1u;
                        levelBytes = levelWidth*levelHeight*levelDepth*nBytesPerTexel;
                        totalBytes += levelBytes;
                        log_console->debugStream() << "[SharedImageResource] \tInitializing texture level " << i << " " <<  utils::toStringVec3(levelWidth, levelHeight, levelDepth) << " with 0's (" << utils::toStringMemory(levelBytes) << ").";

                        glTexSubImage1D(_target, i,
                                0,
                                levelWidth, 
                                validExternalFormat, validExternalType,
                                junkData);
                    }
                    break;


                    //2D TEXTURES
                case(GL_TEXTURE_CUBE_MAP):
                case(GL_TEXTURE_2D):
                case(GL_TEXTURE_1D_ARRAY):
                case(GL_TEXTURE_RECTANGLE):
                    assert(_width >= 1 && _height >= 1 && _depth == 1);
                    glTexStorage2D(_target, _nLevels, _internalFormat, 
                            _width, _height);
                    for (unsigned int i = 0; i < _nLevels; i++) {
                        levelWidth = std::max(1u, _width>>i);
                        levelHeight = std::max(1u, _height>>i);
                        levelDepth = 1u;
                        levelBytes = levelWidth*levelHeight*levelDepth*nBytesPerTexel;
                        totalBytes += levelBytes;
                        log_console->debugStream() << "[SharedImageResource] \tInitializing texture level " << i << " " <<  utils::toStringVec3(levelWidth, levelHeight, levelDepth) << " with 0's (" << utils::toStringMemory(levelBytes) << ").";

                        glTexSubImage2D(_target, i,
                                0,0, 
                                levelWidth, levelHeight, 
                                validExternalFormat, validExternalType,
                                junkData);
                    }
                    break;


                    //3D TEXTURES
                    //case(GL_TEXTURE_CUBE_ARRAY):
                case(GL_TEXTURE_2D_ARRAY):
                case(GL_TEXTURE_3D):
                    assert(_width >= 1 && _height >= 1 && _depth >= 1);
                    glTexStorage3D(_target, _nLevels, _internalFormat, 
                            _width, _height, _depth);
                    for (unsigned int i = 0; i < _nLevels; i++) {

                        levelWidth = std::max(1u, _width>>i);
                        levelHeight = std::max(1u, _height>>i);
                        levelDepth = std::max(1u, _depth>>i);
                        levelBytes = levelWidth*levelHeight*levelDepth*nBytesPerTexel;
                        totalBytes += levelBytes;

                        log_console->debugStream() << "[SharedImageResource] \tInitializing texture level " << i << " " <<  utils::toStringVec3(levelWidth, levelHeight, levelDepth) << " with 0's (" << utils::toStringMemory(levelBytes) << ").";

                        glTexSubImage3D(_target, i,
                                0,0,0, 
                                levelWidth, levelHeight, levelDepth,
                                validExternalFormat, validExternalType,
                                junkData);
                    }
                    break;

                default:
                    log_console->errorStream() << "[SharedImageResource] Texture type not implemented yet !";
                    exit(1);

            }

            log_console->infoStream() << "[SharedImageResource] \tTotal image size : " <<  utils::toStringMemory(totalBytes);

            glBindTexture(_target, 0);

            CHECK_OPENGL_ERRORS();
            assert(glIsTexture(this->getGlResource()));

            log_console->infoStream() << "[SharedImageResource] Registering Image Resource to Cuda..";
            registerResource();

            free(junkData);
        }

            void registerInternalResource() {
                CHECK_CUDA_ERRORS(cudaGraphicsGLRegisterImage(
                            &this->_cudaResource, 
                            this->getGlResource(), 
                            _target,
                            _flags));
            }

            cudaArray_t getCudaArray(unsigned int arrayIndex, unsigned int mipLevel) {
                assert(this->isMapped());
                assert(mipLevel < _nLevels);
               
                switch(_target) {
                    case(GL_TEXTURE_2D_ARRAY):
                    case(GL_TEXTURE_CUBE_MAP):
                        break;
                    default:
                        assert(arrayIndex == 0u);
                }

                cudaArray_t array = 0;
                CHECK_CUDA_ERRORS(cudaGraphicsSubResourceGetMappedArray(&array, this->getCudaResource(), arrayIndex, mipLevel));

                return array;
            }

        private:
            void checkCudaCapabilities() {
                switch(_target) {
                    case(GL_TEXTURE_2D):
                    case(GL_TEXTURE_RECTANGLE):
                    case(GL_TEXTURE_CUBE_MAP):
                    case(GL_TEXTURE_3D):
                    case(GL_TEXTURE_2D_ARRAY):
                    case(GL_RENDERBUFFER):
                        break;
                    default:
                        log_console->errorStream() << "cudaGraphicsGLRegisterImage() does not support target " << utils::toStringTextureTarget(_target) << " yet (18/10/14) !";
                        exit(1);
                }

                switch(_internalFormat) {

                    case(GL_RED):
                    case(GL_RG):
                    case(GL_RGBA):
                    case(GL_LUMINANCE):
                    case(GL_ALPHA):
                    case(GL_LUMINANCE_ALPHA):
                    case(GL_INTENSITY):

                    case(GL_R8):
                    case(GL_R16):
                    case(GL_R16F):
                    case(GL_R32F):
                    case(GL_R8UI):
                    case(GL_R16UI):
                    case(GL_R32UI):
                    case(GL_R8I):
                    case(GL_R16I):
                    case(GL_R32I):

                    case(GL_RG8):
                    case(GL_RG16):
                    case(GL_RG16F):
                    case(GL_RG32F):
                    case(GL_RG8UI):
                    case(GL_RG16UI):
                    case(GL_RG32UI):
                    case(GL_RG8I):
                    case(GL_RG16I):
                    case(GL_RG32I):

                    case(GL_RGBA8):
                    case(GL_RGBA16):
                    case(GL_RGBA16F):
                    case(GL_RGBA32F):
                    case(GL_RGBA8UI):
                    case(GL_RGBA16UI):
                    case(GL_RGBA32UI):
                    case(GL_RGBA8I):
                    case(GL_RGBA16I):
                    case(GL_RGBA32I):

                    case(GL_LUMINANCE8):
                    case(GL_LUMINANCE16):
                    case(GL_LUMINANCE16F_ARB):
                    case(GL_LUMINANCE32F_ARB):
                    case(GL_LUMINANCE8UI_EXT):
                    case(GL_LUMINANCE16UI_EXT):
                    case(GL_LUMINANCE32UI_EXT):
                    case(GL_LUMINANCE8I_EXT):
                    case(GL_LUMINANCE16I_EXT):
                    case(GL_LUMINANCE32I_EXT):

                    case(GL_ALPHA8):
                    case(GL_ALPHA16):
                    case(GL_ALPHA16F_ARB):
                    case(GL_ALPHA32F_ARB):
                    case(GL_ALPHA8UI_EXT):
                    case(GL_ALPHA16UI_EXT):
                    case(GL_ALPHA32UI_EXT):
                    case(GL_ALPHA8I_EXT):
                    case(GL_ALPHA16I_EXT):
                    case(GL_ALPHA32I_EXT):

                    //case(GL_LUMINANCE_ALPHA8):
                    //case(GL_LUMINANCE_ALPHA16):
                    case(GL_LUMINANCE_ALPHA16F_ARB):
                    case(GL_LUMINANCE_ALPHA32F_ARB):
                    case(GL_LUMINANCE_ALPHA8UI_EXT):
                    case(GL_LUMINANCE_ALPHA16UI_EXT):
                    case(GL_LUMINANCE_ALPHA32UI_EXT):
                    case(GL_LUMINANCE_ALPHA8I_EXT):
                    case(GL_LUMINANCE_ALPHA16I_EXT):
                    case(GL_LUMINANCE_ALPHA32I_EXT):

                    case(GL_INTENSITY8):
                    case(GL_INTENSITY16):
                    case(GL_INTENSITY16F_ARB):
                    case(GL_INTENSITY32F_ARB):
                    case(GL_INTENSITY8UI_EXT):
                    case(GL_INTENSITY16UI_EXT):
                    case(GL_INTENSITY32UI_EXT):
                    case(GL_INTENSITY8I_EXT):
                    case(GL_INTENSITY16I_EXT):
                    case(GL_INTENSITY32I_EXT):
                        
                        break;
                    
                    default:
                        log_console->errorStream() << "cudaGraphicsGLRegisterImage() does not support internal format " << utils::toStringInternalFormat(_internalFormat) << " yet (18/10/14) !";
                        exit(1);
                }
            }

        private:
            GLenum _target;
            unsigned int _nLevels;
            unsigned int _internalFormat;
            unsigned int _width, _height, _depth;
            unsigned int _flags;
    };
}

#endif /* end of include guard: CUDAGLSHAREDIMAGERESOURCE_H */
