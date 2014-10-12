
#ifndef PARTICLESYSTEM_H
#define PARTICLESYSTEM_H

#include <vector>

#include "particle.hpp"
#include "renderTree.hpp"
#include "log.hpp"

template<unsigned int N>
class ParticleSystem : public RenderTree {
    public:
        explicit ParticleSystem(unsigned int nParticles_, float dt_);
        virtual ~ParticleSystem();

    protected:
        virtual void initParticles() = 0;
        virtual void updateGrid() = 0;
        virtual void computeAccelerations() = 0;
        virtual void computeSpatialInteractions() = 0;
        virtual void integrateScheme() = 0;

        //renderTree
		virtual void drawDownwards(const float *currentTransformationMatrix = consts::identity4) = 0;
        
        void computeInitialEulerStep();

        std::vector<Particle<N>> _particles;
        
        unsigned int _nParticles;
        float _dt;

    private: 
        void step();
		void animateDownwards() { step(); };
        
};
        
template <unsigned int N>
ParticleSystem<N>::ParticleSystem(unsigned int nParticles_, float dt_) :
    _nParticles(nParticles_),
    _dt(dt_)
{
    log_console->infoStream() << "[ParticleSystem] Creating a new particle system (" 
        << _nParticles << " particles, dt = " << _dt << "s).";
}

template <unsigned int N>
ParticleSystem<N>::~ParticleSystem() 
{
}

template <unsigned int N>
void ParticleSystem<N>::step() 
{
    log_console->debugStream() << "[ParticleSystem] *** Step ***";
    std::cout << _particles[0] << std::endl;
    this->updateGrid();
    this->computeAccelerations();
    this->computeSpatialInteractions();
    this->integrateScheme();
    log_console->debugStream() << "[ParticleSystem] *** End of Step ***";
}

template <unsigned int N>
void ParticleSystem<N>::computeInitialEulerStep() 
{
    log_console->infoStream() << "[ParticleSystem] Computing initial Euler step...";
    for (Particle<N> &p : _particles) {
        for (unsigned int i = 0; i < N; i++) {
            p.v_old()[i] = p.v()[i] - 0.5f*_dt*p.a()[i];
        }
    }
}


#endif /* end of include guard: PARTICLESYSTEM_H */
