
#ifndef BOIDGENERATOR_H
#define BOIDGENERATOR_H

#include "boid.hpp"

class BoidGenerator {
    public:
        virtual ~BoidGenerator() {}

        virtual Boid* operator()();
        virtual const std::string getName() = 0;

    protected:
        BoidGenerator() {};
};

ostream &operator<<(ostream &os, BoidGenerator &bg)
{
	os << bg.getName();	
	return os;
}

#endif /* end of include guard: BOIDGENERATOR_H */
