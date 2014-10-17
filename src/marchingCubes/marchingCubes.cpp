
#include "headers.hpp"
#include <GL/gl.h>
#include <GL/glu.h>
#include "log.hpp"
#include "glUtils.hpp"
#include "utils.hpp"
#include "marchingCubes.hpp"

namespace MarchingCubes {

    MarchingCubes::MarchingCubes(
                    float x0_, float y0_, float z0_,
                    unsigned int W_, unsigned int H_, unsigned int L_,
                    float h_) :
        _x0(x0_), _y0(y0_), _z0(z0_),
        _W(W_), _H(H_), _L(L_),
        _Wf(W_*h_), _Hf(H_*h_), _Lf(L_*h_),
        _h(h_),
        _densitiesPBO(0), _normalsPBO(0),
        _densitiesTexture(0), _normalsTexture(0),
        _densitiesArray(0), _normalsArray(0),
        _densitiesSurface(0), _normalsSurface(0)
    {

        log_console->infoStream() << "[MarchingCube] Created a " << utils::toStringDimension(W_,H_,L_) << " Marching Cube with h=" << h_ << ", real size " << utils::toStringVec3(_Wf,_Hf,_Lf) << ".";

        int cudaDevices[10];
        unsigned int nCudaDevices;
        CHECK_CUDA_ERRORS(cudaGLGetDevices(&nCudaDevices, cudaDevices, 10, cudaGLDeviceListAll));
        log_console->infoStream() << "Found " << nCudaDevices << " CUDA devices corresponding to the current OpenGL context :";
        for (unsigned int i = 0; i < nCudaDevices; i++) {
            log_console->infoStream() << "\tDevice " << cudaDevices[i];
        }
        log_console->infoStream() << "Setting current CUDA device to " << cudaDevices[0] << " !";
        cudaThreadExit();
        cudaSetDevice(cudaDevices[0]);

        allocateAndRegisterTextures();
    }

    MarchingCubes::~MarchingCubes() {

        for (unsigned int i = 0; i < _nRessources; i++) {
            CHECK_CUDA_ERRORS(cudaGraphicsUnregisterResource(_graphicResources[i]));
        }

        delete _densitiesTexture;
        delete _normalsTexture;
    }

    //RenderTree
    void MarchingCubes::animateDownwards() {
        
        static float *vals = new float[_H*_W*_L];
        glBindTexture(GL_TEXTURE_3D, _densitiesTexture->getTextureId());
        glGetTexImage(GL_TEXTURE_3D, 
                0, GL_RED, GL_SHORT, vals);
        for (unsigned int i = 0; i < 10; i++) {
            printf("%f\t", vals[i]);
        }
        printf("\n");
        glBindTexture(GL_TEXTURE_3D, 0);
        CHECK_OPENGL_ERRORS();

        mapTextureResources();
        computeDensities();
        computeTriangles();
        unmapTextureResources();
    }

    void MarchingCubes::drawDownwards(const float *currentTransformationMatrix) {
    }

    //MarchingCubes
    void MarchingCubes::computeDensities() {
        callComputeDensitiesKernel(_x0,_y0,_z0,_W,_H,_L,_h,_densitiesSurface);
    }

    void MarchingCubes::computeTriangles() {
    }

    //CUDA OpenGL
    void MarchingCubes::allocateAndRegisterTextures() {
        //Create textures
        unsigned int texturePBO = 0;
        glGenBuffers(1, &texturePBO);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, texturePBO);
        glBufferData(GL_PIXEL_UNPACK_BUFFER, _W*_H*_L*sizeof(short), NULL, GL_DYNAMIC_DRAW);

        _densitiesTexture = new Texture3D(_W,_H,_L,GL_R16F,0,GL_RED,GL_SHORT);
        _densitiesTexture->addParameter(Parameter(GL_TEXTURE_WRAP_S, GL_CLAMP));
        _densitiesTexture->addParameter(Parameter(GL_TEXTURE_WRAP_T, GL_CLAMP));
        _densitiesTexture->addParameter(Parameter(GL_TEXTURE_WRAP_R, GL_CLAMP));
        _densitiesTexture->addParameter(Parameter(GL_TEXTURE_MAG_FILTER, GL_LINEAR));
        _densitiesTexture->addParameter(Parameter(GL_TEXTURE_MIN_FILTER, GL_LINEAR));
        _densitiesTexture->bindAndApplyParameters(0);
        
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
        CHECK_OPENGL_ERRORS();
        
        _normalsTexture = new Texture3D(_W,_H,_L,GL_RGBA8I,NULL,GL_RGBA_INTEGER,GL_UNSIGNED_B  );
        _normalsTexture->addParameters(_densitiesTexture->getParameters());
        _normalsTexture->bindAndApplyParameters(1);
        CHECK_OPENGL_ERRORS();

        //Register textures
        CHECK_CUDA_ERRORS(
                cudaGraphicsGLRegisterImage(_graphicResources, 
                    _densitiesTexture->getTextureId(),
                    GL_TEXTURE_3D,
                    cudaGraphicsRegisterFlagsSurfaceLoadStore
                    )
                );

        CHECK_CUDA_ERRORS(
                cudaGraphicsGLRegisterImage(_graphicResources + 1, 
                    _normalsTexture->getTextureId(),
                    GL_TEXTURE_3D,
                    cudaGraphicsRegisterFlagsSurfaceLoadStore
                    )
                );
    }


    void MarchingCubes::mapTextureResources() {
        //map ressources
        CHECK_CUDA_ERRORS(cudaGraphicsMapResources(_nRessources, _graphicResources)); //stream

        //get CUDA arrays
        CHECK_CUDA_ERRORS(cudaGraphicsSubResourceGetMappedArray(&_densitiesArray, _graphicResources[0], 0u, 0u));
        CHECK_CUDA_ERRORS(cudaGraphicsSubResourceGetMappedArray(&_normalsArray, _graphicResources[1], 0u, 0u));

        //Create surface objects
        cudaResourceDesc resDesc;
        memset(&resDesc, 0, sizeof(cudaResourceDesc));
        resDesc.resType = cudaResourceTypeArray; 
        resDesc.res.array.array = _densitiesArray;

        CHECK_CUDA_ERRORS(cudaCreateSurfaceObject(&_densitiesSurface, &resDesc));
    }

    void MarchingCubes::unmapTextureResources() {
        CHECK_CUDA_ERRORS(cudaDeviceSynchronize());
        CHECK_CUDA_ERRORS(cudaGetLastError());
        CHECK_CUDA_ERRORS(cudaDestroySurfaceObject(_densitiesSurface));
        CHECK_CUDA_ERRORS(cudaGraphicsUnmapResources(_nRessources, _graphicResources, 0)); //stream
    }

}
