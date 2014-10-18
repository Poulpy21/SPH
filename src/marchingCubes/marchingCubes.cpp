
#include "headers.hpp"
#include "globals.hpp"
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
        _densitiesSurfaceResource(0), _normalsSurfaceResource(0),
        _densitiesSurfaceObject(0), _normalsSurfaceObject(0),
        _trianglesVBO(0), _normalsVBO(0),
        _trianglesProgram("slices")
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

        _densitiesSurfaceResource = new SharedSurfaceResource(GL_TEXTURE_3D, 1u, GL_R32F, _W, _H, _L);
        _normalsSurfaceResource = new SharedSurfaceResource(GL_TEXTURE_3D, 1u, GL_R8, _W, _H, _L);

        mapTextureResources();
        _densitiesSurfaceObject = _densitiesSurfaceResource->createSurfaceObject(0,0);
        _normalsSurfaceObject = _densitiesSurfaceResource->createSurfaceObject(0,0);
        unmapTextureResources();


        //_densitiesTexture->addParameter(Parameter(GL_TEXTURE_WRAP_S, GL_CLAMP));
        //_densitiesTexture->addParameter(Parameter(GL_TEXTURE_WRAP_T, GL_CLAMP));
        //_densitiesTexture->addParameter(Parameter(GL_TEXTURE_WRAP_R, GL_CLAMP));
        //_densitiesTexture->addParameter(Parameter(GL_TEXTURE_MAG_FILTER, GL_LINEAR));
        //_densitiesTexture->addParameter(Parameter(GL_TEXTURE_MIN_FILTER, GL_LINEAR));
        //_densitiesTexture->bindAndApplyParameters(0);

        //translate origin of the renderroot
        this->translate(_x0,_y0,_z0);

        createTriangles();
        makeTrianglesProgram();
    }

    MarchingCubes::~MarchingCubes() {
        CHECK_CUDA_ERRORS(cudaDestroySurfaceObject(_densitiesSurfaceObject));
        CHECK_CUDA_ERRORS(cudaDestroySurfaceObject(_normalsSurfaceObject));
        _densitiesSurfaceObject = 0;
        _normalsSurfaceObject = 0;

    }

    //RenderTree
    void MarchingCubes::animateDownwards() {

        //static float *vals = new float[_H*_W*_L];
        //glBindTexture(GL_TEXTURE_3D, _densitiesSurfaceResource->getGlResource());
        //glGetTexImage(GL_TEXTURE_3D, 
        //0, GL_RED, GL_FLOAT, vals);
        //for (unsigned int i = 0; i < 10; i++) {
        //printf("%f\t", vals[i]);
        //}
        //printf("\n");
        //glBindTexture(GL_TEXTURE_3D, 0);

        mapTextureResources();
        computeDensities();
        computeTriangles();
        unmapTextureResources();
    }

    void MarchingCubes::drawDownwards(const float *currentTransformationMatrix) {

        static float t = 0.0f;
        static bool pos = true;
        if(pos) {
            t += 0.01;
            if(t > 1.0f) {
                pos = false;
                t = 1.0f;
            }
        }
        else {
            t -= 0.01;
            if(t < 0.0f) {
                t = 0.0f;
                pos = true;
            }
        }

        _trianglesProgram.use();
        glBindBufferBase(GL_UNIFORM_BUFFER, 0,  Globals::projectionViewUniformBlock);

        glUniformMatrix4fv(_trianglesUniformLocations["modelMatrix"], 1, GL_TRUE, currentTransformationMatrix);
        glUniform3f(_trianglesUniformLocations["boxSize"], _Wf, _Hf, _Lf);
        glUniform1f(_trianglesUniformLocations["t"], t);
        glUniform1i(_trianglesUniformLocations["surfaceSampler"], 0);

    
        glEnable(GL_TEXTURE_3D);
        glActiveTexture(GL_TEXTURE0);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

        glBindTexture(GL_TEXTURE_3D, _densitiesSurfaceResource->getGlResource());

        glBindBuffer(GL_ARRAY_BUFFER, _trianglesVBO);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
        glEnableVertexAttribArray(0);
        
        glBindBuffer(GL_ARRAY_BUFFER, _normalsVBO);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, 0);
        glEnableVertexAttribArray(1);

        glDrawArrays(GL_TRIANGLES, 0, 6*3);

        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glUseProgram(0);
    }

    //MarchingCubes
    void MarchingCubes::computeDensities() {
        callComputeDensitiesKernel(_x0,_y0,_z0,_W,_H,_L,_h,_densitiesSurfaceObject);
    }

    void MarchingCubes::computeTriangles() {
    }

    void MarchingCubes::mapTextureResources() {
        _densitiesSurfaceResource->mapResource();
        _normalsSurfaceResource->mapResource();
    }

    void MarchingCubes::unmapTextureResources() {
        //CHECK_CUDA_ERRORS(cudaDeviceSynchronize());
        //CHECK_CUDA_ERRORS(cudaGetLastError());

        _densitiesSurfaceResource->unmapResource();
        _normalsSurfaceResource->unmapResource();
    }

    void MarchingCubes::createTriangles() {

        float triangles[6*3*3] = { 
            //plan XY
            0.0f, 0.0f, 0.0f,
            _Wf, 0.0f, 0.0f,
            _Wf, _Hf, 0.0f,

            0.0f, 0.0f, 0.0f,
            _Wf, _Hf, 0.0f,
            0.0f, _Hf, 0.0f,

            //plan XZ
            0.0f, 0.0f, 0.0f,
            _Wf, 0.0f, 0.0f,
            _Wf, 0.0f, _Lf,

            0.0f, 0.0f, 0.0f,
            _Wf, 0.0f, _Lf,
            0.0f, 0.0f, _Lf,

            //plan YZ
            0.0f, 0.0f, 0.0f,
            0.0f, _Hf, 0.0f,
            0.0f, _Hf, _Lf,

            0.0f, 0.0f, 0.0f,
            0.0f, _Hf, _Lf,
            0.0f, 0.0f, _Lf
        };

        float normals[6*3*3] = {
            0.0f, 0.0f, 1.0f,
            0.0f, 0.0f, 1.0f,
            0.0f, 0.0f, 1.0f,
            0.0f, 0.0f, 1.0f,
            0.0f, 0.0f, 1.0f,
            0.0f, 0.0f, 1.0f,

            0.0f, 1.0f, 0.0f,
            0.0f, 1.0f, 0.0f,
            0.0f, 1.0f, 0.0f,
            0.0f, 1.0f, 0.0f,
            0.0f, 1.0f, 0.0f,
            0.0f, 1.0f, 0.0f,

            1.0f, 0.0f, 0.0f,
            1.0f, 0.0f, 0.0f,
            1.0f, 0.0f, 0.0f,
            1.0f, 0.0f, 0.0f,
            1.0f, 0.0f, 0.0f,
            1.0f, 0.0f, 0.0f,
        };

        glGenBuffers(1, &_trianglesVBO);
        glBindBuffer(GL_ARRAY_BUFFER, _trianglesVBO);
        glBufferData(GL_ARRAY_BUFFER, 6*3*3*sizeof(float), triangles, GL_STATIC_DRAW);

        glGenBuffers(1, &_normalsVBO);
        glBindBuffer(GL_ARRAY_BUFFER, _normalsVBO);
        glBufferData(GL_ARRAY_BUFFER, 6*3*3*sizeof(float), normals, GL_STATIC_DRAW);

        glBindBuffer(GL_ARRAY_BUFFER, 0);

        assert(glIsBuffer(_trianglesVBO));
        assert(glIsBuffer(_normalsVBO));
    }

    void MarchingCubes::makeTrianglesProgram() {
        _trianglesProgram.bindAttribLocations("0 1", "pos normals");
        _trianglesProgram.bindFragDataLocation(0, "out_colour");
        _trianglesProgram.bindUniformBufferLocations("0", "projectionView");

        _trianglesProgram.attachShader(Shader("shaders/marchingCubes/slice.vs", GL_VERTEX_SHADER));
        _trianglesProgram.attachShader(Shader("shaders/marchingCubes/slice.fs", GL_FRAGMENT_SHADER));

        _trianglesProgram.link();

        _trianglesUniformLocations = _trianglesProgram.getUniformLocationsMap("modelMatrix boxSize t surfaceSampler", false);
    }

}
