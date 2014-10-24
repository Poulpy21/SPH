
#ifndef BOIDINTERACTION_H
#define BOIDINTERACTION_H

#include <list>
#include "boid.hpp"
#include "neighborStruct.hpp"

class BoidInteraction {
public:
    virtual ~BoidInteraction() {};
    
    float computeForce(Boid* boid, NeighborStruct &neighborStruct) {
        return computeInteration(boid, neighborStruct.getNearbyNeighbors(boid->x, _r_min, _r_max));
    }
    
protected:
    BoidInteraction(float r_min, float r_max) :
        _r_min(r_min), _r_max(r_max) 
    {
    }
    
    virtual float computeInteration(Boid *boid, std::list<Boid*> neighbors) = 0;

private:
    float _r_min, _r_max;
};



#endif /* end of include guard: BOIDINTERACTION_H */
