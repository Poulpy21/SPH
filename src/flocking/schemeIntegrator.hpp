
#ifndef SHEMEINTEGRATOR_H
#define SHEMEINTEGRATOR_H

#include <list>
#include <string>
#include "boid.hpp"

class SchemeIntegrator {
    public:
        virtual ~SchemeIntegrator() {}
        virtual void integrateScheme(std::list<Boid*> boids) = 0;
        virtual const std::string getName() = 0;

    protected:
        SchemeIntegrator() {};
};

ostream &operator<<(ostream &os, SchemeIntegrator &si)
{
	os << si.getName();	
	return os;
}


#endif /* end of include guard: SHEMEINTEGRATOR_H */
