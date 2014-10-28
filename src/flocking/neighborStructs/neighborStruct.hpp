
#ifndef NEIGHBORSTRUCT_H
#define NEIGHBORSTRUCT_H

#include <list>
#include "boid.hpp"

template <typename S>
class NeighborStruct {
public:
    virtual ~NeighborStruct() {}
    
    virtual void insertBoid(Boid<S> *boid) = 0;
    virtual void removeBoid(unsigned int boidId, Vec helperPosition) {
        throw new std::runtime_error("Boid removal is not supported in this neighbor structure !");
    }
    
	virtual std::list<Boid<S>*> getBoids() = 0;
    virtual std::list<Boid<S>*> getNearbyNeighbors(Vec pos, float min_radius, float max_radius) = 0;

	virtual void update() = 0;

	virtual const std::string getName() = 0;

protected:
    NeighborStruct() {}
};

template <typename S>
ostream &operator<<(ostream &os, NeighborStruct<S> &ns)
{
	os << ns.getName();	
	return os;
}


#endif /* end of include guard: NEIGHBORSTRUCT_H */
