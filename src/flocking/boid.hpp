
#ifndef BOID_H
#define BOID_H

#include "utils.hpp"

using namespace qglviewer;

struct Boid {
        Boid(Vec x0, Vec v0, Vec a0, float m);
        virtual ~Boid ();
        
        static unsigned int globalId;
        
        unsigned int id;
        Vec x, v, a;
        Vec v_old;

        float m;
};
        
unsigned int Boid::globalId = 0;

std::ostream & operator << (std::ostream &os, Boid &B) {
    os << "Particule" << std::endl;
    os << "\tm = " << B.m;
    os << "\tX = " << B.x;
    os << "\tV = " << B.v;
    os << "\tA = " << B.a;
    os << "\tVold = " << B.v_old;
    return os;
}

Boid::Boid(Vec x0, Vec v0, Vec a0, float m0) :
id(0), x(x0), v(v0), a(a0), m(m0) {
            //compute initial Euler step with the given dt
            //v_old=  v - 0.5f*dt*a;
            //TODO

            id = Boid::globalId;
            Boid::globalId++;
}

Boid::~Boid() {
}

#endif /* end of include guard: BOID_H */
