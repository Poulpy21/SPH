
#ifndef UNIFORMBOIDGENERATOR_H
#define UNIFORMBOIDGENERATOR_H

#include "boid.hpp"
#include "rand.hpp"

class UniformBoidGenerator {
    public:
        UniformBoidGenerator(qglviewer::vec xmin, qglviewer xmax, float M_min, float M_max);
        ~UniformBoidGenerator();

        Boid* operator()();

    private:
        qglviewer::vec _xmin, _xmax;
        float _M_min, _M_max
}
        
UniformBoidGenerator(qglviewer::vec xmin, qglviewer xmax, float M_min, float M_max) :
    _xmin(xmin), _xmax(xmax), _M_min(M_min), _M_max(M_max)
{
}

~UniformBoidGenerator() {}

Boid* operator()() {
    qglviewer::Vec zero(0,0,0);
    qglviewer::Vec x0 = Random::randPos(_xmin, _xmax);
    float m = Random::randf(_M_min, _M_max);
    return new Boid(x0, zero, zero, m);
}

#endif /* end of include guard: UNIFORMBOIDGENERATOR_H */
