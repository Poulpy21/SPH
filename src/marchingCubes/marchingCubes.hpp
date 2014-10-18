
#ifndef MARCHINGCUBES_H
#define MARCHINGCUBES_H

#include "headers.hpp"
#include "defines.hpp"
#include "renderRoot.hpp"
#include "texture3D.hpp"

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
            void allocateAndRegisterTextures();
            void mapTextureResources();
            void unmapTextureResources();

            //VARS
            float _x0, _y0, _z0;     // where does the domain begin
            unsigned int _W, _H, _L; // size of the domain (uint)
            float _Wf, _Hf, _Lf;     // size of the domain (float)
            float _h;                // space delta


            //3D Textures
            SharedSurfaceResource *densities, *textures;

            static const unsigned int _nRessources = 2;
            cudaGraphicsResource_t _graphicResources[_nRessources];
    };

    //CUDA
    extern void bindSurfaces(const cudaArray_t densitiesArray, const cudaArray_t normalsArray);
    
    extern void callComputeDensitiesKernel(float x0, float y0, float z0,
            unsigned int W, unsigned int H, unsigned int L, 
            float h, 
            cudaSurfaceObject_t densitiesSurface);

    //extern void computeTrianglesKernel();
}

#endif /* end of include guard: MARCHINGCUBES_H */
