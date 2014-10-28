
#ifndef BOID_H
#define BOID_H

#include "utils.hpp"

using namespace qglviewer;

template <typename S>
struct Boid {
        Boid(Vec x0, Vec v0, Vec a0, float m);
        virtual ~Boid ();
        
        static unsigned int globalId;
        
        unsigned int id;
        float m;

        Vec x, v, a;

        S scheme;
};
        
template <typename S>
unsigned int Boid<S>::globalId = 0;

template <typename S>
std::ostream & operator << (std::ostream &os, Boid<S> &B) {
    os << "Particule" << std::endl;
    os << "\tm = " << B.m;
    os << "\tX = " << B.x;
    os << "\tV = " << B.v;
    os << "\tA = " << B.a;
    os << "\t" << B.scheme;
    return os;
}

template <typename S>
Boid<S>::Boid(Vec x0, Vec v0, Vec a0, float m0) :
id(0), x(x0), v(v0), a(a0), m(m0) {
            id = Boid<S>::globalId;
            Boid<S>::globalId++;
}
            
template <typename S>
Boid<S>::~Boid() {
}

//compute initial Euler step with the given dt
//v_old=  v - 0.5f*dt*a;

#endif /* end of include guard: BOID_H */
