
#ifndef BOIDINTERACTION_H
#define BOIDINTERACTION_H

#include <list>
#include <string>
#include "boid.hpp"
#include "neighborStruct.hpp"

template <typename S>
class BoidInteraction {
public:
    virtual ~BoidInteraction() {};
    virtual float computeInteration(NeighborStruct<S> *neighborStruct) = 0;
	virtual const std::string getName() = 0;
    
protected:
    BoidInteraction(float r_min, float r_max) :
        _r_min(r_min), _r_max(r_max) 
    {
    }
    

private:
    float _r_min, _r_max;
};

template <typename S>
ostream &operator<<(ostream &os, BoidInteraction<S> &bi)
{
	os << bi.getName();	
	return os;
}

#endif /* end of include guard: BOIDINTERACTION_H */
