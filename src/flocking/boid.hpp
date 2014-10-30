
#ifndef BOID_H
#define BOID_H

#include <iostream>
#include "vec.hpp"
#include "defines.hpp"

#ifdef __CUDACC__
#define __HOST__ __host__
#define __DEVICE__ __device__
#else
#define __HOST__
#define __DEVICE__
#endif 
        
struct ALIGN(8) Boid {
        unsigned int id;
        float m;
        Vec x, v, a;
        
        __HOST__ __DEVICE__ Boid(Vec x0, Vec v0, Vec a0, float m);
        __HOST__ __DEVICE__ Boid(const Boid &b);
        __HOST__ __DEVICE__ Boid & operator= (const Boid &b);
        __HOST__ __DEVICE__ ~Boid();
};


__HOST__ std::ostream & operator << (std::ostream &os, Boid &B);
        
#undef __HOST__
#undef __DEVICE__

#endif /* end of include guard: BOID_H */
