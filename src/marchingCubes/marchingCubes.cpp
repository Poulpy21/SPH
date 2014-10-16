
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
        _densities(0), _normals(0),
        _densitiesArray(0), _normalsArray(0)
    {

        log_console->infoStream() << "[MarchingCube] Created a " << utils::toStringDimension(W_,H_,L_) << " Marching Cube with h=" << h_ << ", real size " << utils::toStringVec3(_Wf,_Hf,_Lf) << ".";

        allocateAndRegisterTextures();
    }

    MarchingCubes::~MarchingCubes() {

        for (unsigned int i = 0; i < _nRessources; i++) {
            CHECK_CUDA_ERRORS(cudaGraphicsUnregisterResource(_graphicResources[i]));
        }

        delete _densities;
        delete _normals;
    }

    //RenderTree
    void MarchingCubes::animateDownwards() {
        
        float *vals = new float[10];
        _densities->bindAndApplyParameters(0);
        glGetnTexImage(GL_TEXTURE_3D, 
                0, GL_RED, GL_FLOAT, 10, vals);
        for (unsigned int i = 0; i < 10; i++) {
            printf("%f\t", vals[i]);
        }
        printf("\n");

        mapTextureResources();
        computeDensities();
        computeTriangles();
        unmapTextureResources();
    }

    void MarchingCubes::drawDownwards(const float *currentTransformationMatrix) {
    }

    //MarchingCubes
    void MarchingCubes::computeDensities() {
        callComputeDensitiesKernel(_x0,_y0,_z0,_W,_H,_L,_h);
    }

    void MarchingCubes::computeTriangles() {
    }

    //CUDA OpenGL
    void MarchingCubes::allocateAndRegisterTextures() {

        _densities = new Texture3D(_W,_H,_L,GL_RED,NULL,GL_RED,GL_FLOAT);
        _densities->addParameter(Parameter(GL_TEXTURE_WRAP_S, GL_CLAMP));
        _densities->addParameter(Parameter(GL_TEXTURE_WRAP_T, GL_CLAMP));
        _densities->addParameter(Parameter(GL_TEXTURE_WRAP_R, GL_CLAMP));
        _densities->addParameter(Parameter(GL_TEXTURE_MAG_FILTER, GL_LINEAR));
        _densities->addParameter(Parameter(GL_TEXTURE_MIN_FILTER, GL_LINEAR));
        _densities->bindAndApplyParameters(0);
        CHECK_OPENGL_ERRORS();

        _normals = new Texture3D(_W,_H,_L,GL_RGBA32F,NULL,GL_RGBA,GL_FLOAT);
        _normals->addParameters(_densities->getParameters());
        _normals->bindAndApplyParameters(1);
        CHECK_OPENGL_ERRORS();

        log_console->infoStream() << _densities->getTextureId() << " " << _normals->getTextureId();

        CHECK_CUDA_ERRORS(
                cudaGraphicsGLRegisterImage(_graphicResources, 
                    _densities->getTextureId(),
                    GL_TEXTURE_3D,
                    cudaGraphicsRegisterFlagsSurfaceLoadStore
                    )
                );

        CHECK_CUDA_ERRORS(
                cudaGraphicsGLRegisterImage(_graphicResources + 1, 
                    _normals->getTextureId(),
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

        //bind CUDA surfaces to CUDA arrays
        bindSurfaces(_densitiesArray, _normalsArray);
    }

    void MarchingCubes::unmapTextureResources() {
        CHECK_CUDA_ERRORS(cudaGraphicsUnmapResources(_nRessources, _graphicResources, 0)); //stream
    }

}
