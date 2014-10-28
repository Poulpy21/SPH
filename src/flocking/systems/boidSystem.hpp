
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

template <typename S>
class BoidSystem : public RenderTree {
    public:
        explicit BoidSystem(float dt);
        virtual ~BoidSystem();
        
        //Attach a neighbor helper structure to this boid system 
		void attachNeighborStruct(NeighborStruct<S> *neighborStruct, bool keepAlive=false);
        
        //Attach a scheme integration method to this boid system 
		void attachSchemeIntegrator(SchemeIntegrator<S> *schemeIntegrator, bool keepAlive=false);
    
        //Add an interaction between boids
		void addInteraction(BoidInteraction<S> *interaction);
        
        //Initialize numerical scheme (after all boids have been generated)
        //After this call, numerical simulation can begin.
        void initScheme();

    protected:
		
        //Boid system//
        // 
        //Generate 'nBoids' boids with the given generator 'boidGenerator'.
        void generateBoids(BoidGenerator<S> &boidGenerator, unsigned int nBoids); 
        // 
        //Update attached neighbor structure
        void updateNeighborStructure(); 
        //
        //Compute forces applied to boids (modifies their acceleration)
        void computeForces();
        //
        //Integrate numerical scheme with attached integration scheme
        void integrateScheme(); 
        

        //RenderTree//
        //
        //Draw the BoidSystem to screen (called by the QGLViewer).
        //By default we draw nothing and we just simulate the system.
		virtual void drawDownwards(const float *currentTransformationMatrix = consts::identity4); 
	
		//Boid system//
        
		NeighborStruct<S>* _neighborStruct;            //Attached helper structure
		SchemeIntegrator<S>* _schemeIntegrator;        //Attached numerical scheme
        std::list<BoidInteraction<S>*> _interactions;  //Interactions between boids
        
        unsigned int _nBoids; //Current number of generated boids
        float _dt;            //Time step used for the numerical scheme
        
        //Should these attached structures be deleted at class destruction ?
		bool _keepBoidNeighborStructAlive, _keepBoidSchemeIntegratorAlive;
        bool _schemeInitialized;

    private: 

        //This is called by QGLViewer before each rendered OpenGL frame.
		void animateDownwards() { step(); };
        
        //Compute one step of the simulation.
        void step();
        
};
        
template <typename S>
BoidSystem<S>::BoidSystem(float dt_) :
	_neighborStruct(0),
	_interactions(),
    _nBoids(0),
    _dt(dt_),
	_keepBoidNeighborStructAlive(false),
	_keepBoidSchemeIntegratorAlive(false),
    _schemeInitialized(false)
{
    log_console->infoStream() << "[BoidSystem] Creating a new boid system (" 
        << _nBoids << " boids, dt = " << _dt << "s).";
}

template <typename S>
BoidSystem<S>::~BoidSystem() 
{
	if(!_keepBoidNeighborStructAlive)
		delete _neighborStruct;
	if(!_keepBoidSchemeIntegratorAlive)
		delete _schemeIntegrator;
}
		
template <typename S>
void BoidSystem<S>::attachSchemeIntegrator(SchemeIntegrator<S> *schemeIntegrator, bool keepAlive) {
	assert(schemeIntegrator != 0);
	if(!_keepBoidSchemeIntegratorAlive && _schemeIntegrator)
		delete _schemeIntegrator;
    log_console->infoStream() << "[BoidSystem] Attaching scheme integrator " << schemeIntegrator->getName() << " !";
	_schemeIntegrator = schemeIntegrator;
	_keepBoidSchemeIntegratorAlive = keepAlive;
}

template <typename S>
void BoidSystem<S>::attachNeighborStruct(NeighborStruct<S> *neighborStruct, bool keepAlive) {
    log_console->infoStream() << "[BoidSystem] Attaching neighbor structure " << neighborStruct->getName() << " !";
	assert(neighborStruct != 0);
	if(!_keepBoidNeighborStructAlive && _neighborStruct)
		delete _neighborStruct;
	_neighborStruct = neighborStruct;
	_keepBoidNeighborStructAlive = keepAlive;
}
        
template <typename S>
void BoidSystem<S>::addInteraction(BoidInteraction<S> *interaction) {
    log_console->infoStream() << "[BoidSystem] Adding interaction " << interaction->getName() << " !";
	_interactions.push_back(interaction);
}
        
template <typename S>
void BoidSystem<S>::initScheme() {
    assert(_schemeIntegrator != 0);
    _schemeIntegrator->initScheme(_neighborStruct->getBoids());
    _schemeInitialized = true;
}

template <typename S>
void BoidSystem<S>::step() 
{
    if(!_schemeInitialized)
        return;

    assert(_neighborStruct != 0);
    log_console->debugStream() << "[BoidSystem] *** Step ***";
    this->updateNeighborStructure();
    this->computeForces();
    this->integrateScheme();
    log_console->debugStream() << "[BoidSystem] *** End of Step ***";
}
        
template <typename S>
void BoidSystem<S>::generateBoids(BoidGenerator<S> &boidGenerator, unsigned int nBoids) {
	log_console->debugStream() << "[BoidSystem] Generating " << _nBoids << " boids with generator " << boidGenerator.getName() << " !";
    assert(_neighborStruct != 0);
	for (unsigned int i = 0; i < nBoids; i++) {
		_neighborStruct->insertBoid(boidGenerator());
	}
    _nBoids += nBoids;
}

template <typename S>
void BoidSystem<S>::updateNeighborStructure() {
	log_console->debugStream() << "[BoidSystem] Updating neighbor structure !";
    assert(_neighborStruct != 0);
	_neighborStruct->update();
}
        
template <typename S>
void BoidSystem<S>::computeForces() {
	log_console->debugStream() << "[BoidSystem] Computing forces !";
	for(BoidInteraction<S> *bi : _interactions) {
		bi->computeInteration(_neighborStruct);
	}
}
        
template <typename S>
void BoidSystem<S>::integrateScheme() {
	log_console->debugStream() << "[BoidSystem] Integrating scheme !";
    assert(_schemeIntegrator != 0);
    _schemeIntegrator->integrateScheme(_neighborStruct->getBoids());
}
		
template <typename S>
void BoidSystem<S>::drawDownwards(const float *currentTransformationMatrix) {
	log_console->debugStream() << "[BoidSystem] There is nothing to draw !";
}

#endif /* end of include guard: BOIDSYSTEM_H */
