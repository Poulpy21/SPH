
#ifndef SIMPLEPARTICLESYSTEM2D_H
#define SIMPLEPARTICLESYSTEM2D_H

#include "headers.hpp"
#include "globals.hpp"
#include "program.hpp"
#include "particleSystem.hpp"
#include "particle.hpp"

#include <map>

class SimpleParticleSystem2D : public ParticleSystem<2u> {
    
    public: 
        explicit SimpleParticleSystem2D(unsigned int nParticles_, float dt_);
        ~SimpleParticleSystem2D();
    
    private:
        void initParticles();
        void updateGrid();
        void computeAccelerations();
        void computeSpatialInteractions();
        void integrateScheme();		
        void drawDownwards(const float *currentTransformationMatrix = consts::identity4);
        void drawParticles(const float *currentTransformationMatrix = consts::identity4);
        void drawDomain(const float *currentTransformationMatrix = consts::identity4);

        const float _dh    = 0.068f;  // smoothing length
        const float _m     = 0.0033f; // particle mass
        const float _rho_0 = 2.861f;  // reference density
        const float _P_0   = 0.5;     // pressure constant
        const float _g     = 9.81f;   // acceleration of gravity
        const float _nu    = 0.01f;   // dynamic viscosity
        const float _k_0   = 8;       // initial motion damping
        const float _k     = 1;       // motion damping

        const float _w = 2.0f;
        const float _h = 1.0f;
       
        //OpenGL
        Program _particleProgram, _domainProgram;
        std::map<std::string, int> _particleUniformLocations;
        std::map<std::string, int> _domainUniformLocations;

        GLuint _particleVBO, _domainVBO;
        GLfloat *_particlePositions;

        //SPH
        float cubicSpline(float q);
        float D_cubicSpline(float q);
        float D2_cubicSpline(float q);

        float W(float* xi, float* xj);
        float D_Wx(float* xi, float* xj);
        float D_Wy(float* xi, float* xj);
        float D2_W(float* xi, float* xj);

        //Grid
        std::vector<Particle<2u>*> **_neighborGrid;
        unsigned int _gridWidth, _gridHeight;

        void makeParticleProgram();
        void makeDomainProgram();
        void makeGrid();
};


#endif /* end of include guard: SIMPLEPARTICLESYSTEM2D_H */
