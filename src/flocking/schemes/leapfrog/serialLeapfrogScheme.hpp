
#ifndef LEAPFROGSERIAL_H
#define LEAPFROGSERIAL_H

#include "leapfrogScheme.hpp"


class SerialLeapFrogScheme : public LeapFrogScheme {
    
    using S = LeapfrogStruct;

    public:
        SerialLeapFrogScheme(float dt) : LeapFrogScheme(dt) {};
        ~SerialLeapFrogScheme() {};

        void initScheme(std::list<Boid*> boids) {
            //for(Boid* B : boids) {
                //B->scheme.v_old =  B->v - 0.5f*_dt*B->a;
            //}
        }

        void integrateScheme(std::list<Boid*> boids) {
            Vec v_h;
            //for(Boid* B : boids) {
                //v_h = B->scheme.v_old + B->a*_dt;
                //B->x = B->x + v_h*_dt;
                //B->v = 0.5*(B->v_old + v_h);
                //B->scheme.v_old = v_h;
            //}
        }
        
        const std::string getName() {
            return "Leapfrog (serial implementation)";
        }
};


#endif /* end of include guard: LEAPFROGSERIAL_H */
