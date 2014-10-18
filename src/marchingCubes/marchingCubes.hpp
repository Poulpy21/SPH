
#ifndef MARCHINGCUBES_H
#define MARCHINGCUBES_H

#include "headers.hpp"
#include "defines.hpp"
#include "renderRoot.hpp"
#include "sharedSurfaceResource.hpp"

#include "program.hpp"

using namespace cuda_gl_interop;

namespace MarchingCubes {

    class MarchingCubes : public RenderTree {

        public:
            explicit MarchingCubes(
                    float x0, float y0, float z0,
                    unsigned int W, unsigned int H, unsigned int L,
                    float h);

            virtual ~MarchingCubes();

        private:

            //RenderTree
            void animateDownwards();
            void drawDownwards(const float *currentTransformationMatrix = consts::identity4);

            //MarchingCubes
            void computeDensities();
            void computeTriangles();

            //OpenGL - Cuda
            void mapTextureResources();
            void unmapTextureResources();

            //OpenGL
            void createTriangles();
            void makeTrianglesProgram();

            //VARS
            float _x0, _y0, _z0;     // where does the domain begin
            unsigned int _W, _H, _L; // size of the domain (uint)
            float _Wf, _Hf, _Lf;     // size of the domain (float)
            float _h;                // space delta
            
            //3D Surfaces
            SharedSurfaceResource *_densitiesSurfaceResource, *_normalsSurfaceResource;
            cudaSurfaceObject_t _densitiesSurfaceObject, _normalsSurfaceObject;

            //VBOs
            unsigned int _trianglesVBO, _normalsVBO;
            Program _trianglesProgram;
            std::map<std::string, int> _trianglesUniformLocations;


    };

    //CUDA
    extern void callComputeDensitiesKernel(
            float x0, float y0, float z0,
            unsigned int W, unsigned int H, unsigned int L, 
            float h, 
            cudaSurfaceObject_t densitiesSurface);

    //extern void computeTrianglesKernel();
}

#endif /* end of include guard: MARCHINGCUBES_H */
