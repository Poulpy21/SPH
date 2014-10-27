
#ifndef BOIDSYSTEM_H
#define BOIDSYSTEM_H

#include <vector>

#include "boid.hpp"
#include "boidGenerator.hpp"
#include "boidInteraction.hpp"
#include "neighborStruct.hpp"
#include "schemeIntegrator.hpp"
#include "renderTree.hpp"
#include "log.hpp"

class BoidSystem : public RenderTree {
    public:
        explicit BoidSystem(unsigned int nBoids, float dt);
        virtual ~BoidSystem();

		void attachNeighborStruct(NeighborStruct *neighborStruct, bool keepAlive=false);
		void attachSchemeIntegrator(SchemeIntegrator *schemeIntegrator, bool keepAlive=false);

		void addInteraction(BoidInteraction *interaction);

    protected:
        void initBoids(BoidGenerator &boidGenerator);
        void updateNeighborStructure();
        void computeForces();
        void integrateScheme();

        //renderTree
		virtual void drawDownwards(const float *currentTransformationMatrix = consts::identity4) = 0;
	
		//Boid system
		NeighborStruct* _neighborStruct;
		SchemeIntegrator* _schemeIntegrator; 
        std::list<BoidInteraction*> _interactions;
        
        unsigned int _nBoids;
        float _dt;

		bool _keepBoidNeighborStructAlive, _keepBoidSchemeIntegratorAlive;

    private: 
        void step();
		void animateDownwards() { step(); };
        
};
        
BoidSystem::BoidSystem(unsigned int nBoids_, float dt_) :
	_neighborStruct(0),
	_interactions(),
    _nBoids(nBoids_),
    _dt(dt_),
	_keepBoidNeighborStructAlive(false),
	_keepBoidSchemeIntegratorAlive(false)
{
    log_console->infoStream() << "[BoidSystem] Creating a new boid system (" 
        << _nBoids << " boids, dt = " << _dt << "s).";
}

BoidSystem::~BoidSystem() 
{
	if(!_keepBoidNeighborStructAlive)
		delete _neighborStruct;
	if(!_keepBoidSchemeIntegratorAlive)
		delete _schemeIntegrator;
}
		
void BoidSystem::attachSchemeIntegrator(SchemeIntegrator *schemeIntegrator, bool keepAlive) {
	assert(schemeIntegrator != 0);
	if(!_keepBoidSchemeIntegratorAlive && _schemeIntegrator)
		delete _schemeIntegrator;
    log_console->infoStream() << "[BoidSystem] Attaching scheme integrator " << schemeIntegrator->getName() << " !";
	_schemeIntegrator = schemeIntegrator;
	_keepBoidSchemeIntegratorAlive = keepAlive;
}

void BoidSystem::attachNeighborStruct(NeighborStruct *neighborStruct, bool keepAlive) {
	assert(neighborStruct != 0);
	if(!_keepBoidNeighborStructAlive && _neighborStruct)
		delete _neighborStruct;
    log_console->infoStream() << "[BoidSystem] Attaching neighbor structure " << neighborStruct->getName() << " !";
	_neighborStruct = neighborStruct;
	_keepBoidNeighborStructAlive = keepAlive;
}
        
void BoidSystem::addInteraction(BoidInteraction *interaction) {
    log_console->infoStream() << "[BoidSystem] Adding interaction " << interaction->getName() << " !";
	_interactions.push_back(interaction);
}

void BoidSystem::step() 
{
    log_console->debugStream() << "[BoidSystem] *** Step ***";
    this->updateNeighborStructure();
    this->computeForces();
    this->integrateScheme();
    log_console->debugStream() << "[BoidSystem] *** End of Step ***";
}
        
void BoidSystem::initBoids(BoidGenerator &boidGenerator) {
	log_console->debugStream() << "[BoidSystem] Initializing " << _nBoids << " boids !";
	for (unsigned int i = 0; i < _nBoids; i++) {
		_neighborStruct->insertBoid(boidGenerator());
	}
}

void BoidSystem::updateNeighborStructure() {
	log_console->debugStream() << "[BoidSystem] Updating neighbor structure !";
	_neighborStruct->update();
}
        
void BoidSystem::computeForces() {
	log_console->debugStream() << "[BoidSystem] Computing forces !";
	for(BoidInteraction *bi : _interactions) {
		bi->computeInteration(_neighborStruct);
	}
}
        
void BoidSystem::integrateScheme() {
	log_console->debugStream() << "[BoidSystem] Integrating scheme !";
    _schemeIntegrator->integrateScheme(_neighborStruct->getBoids());
}

#endif /* end of include guard: BOIDSYSTEM_H */
