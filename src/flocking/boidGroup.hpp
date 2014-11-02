
#ifndef BOIDGROUP_H
#define BOIDGROUP_H

#include "boid.hpp"

#ifdef __CUDACC__
#define __HOST__ __host__
#define __DEVICE__ __device__
#else
#define __HOST__
#define __DEVICE__
#endif 

struct BoidGroup {
    const char* groupName;
    unsigned int nBoids;
    Boid *boids;
    
    __HOST__ __DEVICE__ BoidGroup();
    __HOST__ __DEVICE__ BoidGroup(const char* groupName, Boid *boids, unsigned int nBoids);
    __HOST__ __DEVICE__ BoidGroup(const BoidGroup &bg);
    __HOST__ __DEVICE__ ~BoidGroup();
    
    __HOST__ __DEVICE__ BoidGroup& operator= (const BoidGroup &bg);

    __HOST__ __DEVICE__ unsigned long long int hashName();
        
    //groupName(groupName), nBoids(nBoids), boids(boids) { }
};

#undef __HOST__
#undef __DEVICE__

#endif /* end of include guard: BOIDGROUP_H */
