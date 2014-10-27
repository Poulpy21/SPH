
#ifndef NEIGHBORSTRUCT_H
#define NEIGHBORSTRUCT_H

#include <list>
#include "boid.hpp"

class NeighborStruct {
public:
    virtual ~NeighborStruct() {}
    
    virtual void insertBoid(Boid *boid) = 0;
    virtual void removeBoid(unsigned int boidId, Vec helperPosition) {
        throw new std::runtime_error("Boid removal is not supported in this neighbor structure !");
    }
    
	virtual std::list<Boid*> getBoids() = 0;
    virtual std::list<Boid*> getNearbyNeighbors(Vec pos, float min_radius, float max_radius) = 0;

	virtual void update() = 0;

	virtual const std::string getName() = 0;

protected:
    NeighborStruct() {}
};

ostream &operator<<(ostream &os, NeighborStruct &ns)
{
	os << ns.getName();	
	return os;
}


#endif /* end of include guard: NEIGHBORSTRUCT_H */
