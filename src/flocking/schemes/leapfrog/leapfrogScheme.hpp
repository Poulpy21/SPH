
#ifndef LEAPFROG_H
#define LEAPFROG_H

#include "schemeIntegrator.hpp"

using namespace qglviewer;

struct LeapfrogStruct {
    Vec v_old;
};

class LeapFrogScheme : public SchemeIntegrator<LeapfrogStruct> {
    public:
        virtual ~LeapFrogScheme() {};
    protected:
        LeapFrogScheme(float dt) : SchemeIntegrator<LeapfrogStruct>(dt) {};
};

#endif /* end of include guard: LEAPFROG_H */
