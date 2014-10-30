
#ifndef LEAPFROG_H
#define LEAPFROG_H

#include "schemeIntegrator.hpp"


struct LeapfrogStruct {
    Vec *v_old;
};

class LeapFrogScheme : public SchemeIntegrator {
    public:
        virtual ~LeapFrogScheme() {};
    protected:
        LeapFrogScheme(float dt) : SchemeIntegrator(dt) {};
};

#endif /* end of include guard: LEAPFROG_H */
