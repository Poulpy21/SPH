
#ifndef BOIDSYSTEM_H
#define BOIDSYSTEM_H

#include <vector>

#include "boid.hpp"
#include "boidGenerator.hpp"
#include "boidInteraction.hpp"
#include "neighborStruct.hpp"
#include "renderTree.hpp"
#include "log.hpp"

class BoidSystem : public RenderTree {
    public:
        explicit BoidSystem(unsigned int nBoids, float dt);
        virtual ~BoidSystem();

        void addInteraction(BoidInteraction *interaction);

    protected:
        void initBoids(BoidGenerator & boidGenerator);
        void updateNeighborGrid();
        void computeForces();
        void updateGrid();

        virtual void integrateScheme() = 0;

        //renderTree
		virtual void drawDownwards(const float *currentTransformationMatrix = consts::identity4) = 0;

        std::list<Boid*> _boids;
        std::list<BoidInteraction*> _interactions;
        
        unsigned int _nBoids;
        float _dt;

    private: 
        void step();
		void animateDownwards() { step(); };
        
};
        
BoidSystem::BoidSystem(unsigned int nBoids_, float dt_) :
    _nBoids(nBoids_),
    _dt(dt_)
{
    log_console->infoStream() << "[BoidSystem] Creating a new boid system (" 
        << _nBoids << " boids, dt = " << _dt << "s).";
}

BoidSystem::~BoidSystem() 
{
}

void BoidSystem::step() 
{
    log_console->debugStream() << "[BoidSystem] *** Step ***";
    this->updateGrid();
    this->computeForces();
    //this->computeSpatialInteractions();
    this->integrateScheme();
    log_console->debugStream() << "[BoidSystem] *** End of Step ***";
}


#endif /* end of include guard: BOIDSYSTEM_H */
