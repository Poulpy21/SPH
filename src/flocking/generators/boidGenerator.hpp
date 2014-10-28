
#ifndef BOIDGENERATOR_H
#define BOIDGENERATOR_H

#include "boid.hpp"

template <typename S>
class BoidGenerator {
    public:
        virtual ~BoidGenerator() {}

        virtual Boid<S>* operator()();
        virtual const std::string getName() = 0;

    protected:
        BoidGenerator() {};
};

template <typename S>
ostream &operator<<(ostream &os, BoidGenerator<S> &bg)
{
	os << bg.getName();	
	return os;
}

#endif /* end of include guard: BOIDGENERATOR_H */
