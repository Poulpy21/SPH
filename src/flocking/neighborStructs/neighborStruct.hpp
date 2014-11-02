
#ifndef NEIGHBORSTRUCT_H
#define NEIGHBORSTRUCT_H

#include <list>
#include <iostream>

#include "boid.hpp"
#include "boidGroup.hpp"
#include "simplyLinkedList.hpp"

#ifdef __CUDACC__
#define __HOST__ __host__
#define __DEVICE__ __device__
#else
#define __HOST__
#define __DEVICE__
#endif 

class NeighborStruct {
    public:
        __HOST__ __DEVICE__ NeighborStruct();
        __HOST__ __DEVICE__ NeighborStruct(const NeighborStruct &ns);
        __HOST__ __DEVICE__ NeighborStruct& operator= (const NeighborStruct& ns); 
        __HOST__ __DEVICE__ virtual ~NeighborStruct();

        __HOST__ __DEVICE__ BoidGroup* getBoidGroup(const char* groupName);
        __HOST__ __DEVICE__ void insertBoidGroup(BoidGroup *boidGroup);
        __HOST__ __DEVICE__ void removeBoidGroup(const char* groupName);

        __HOST__ __DEVICE__ virtual List<Boid> getNearbyNeighbors(const char* groupName, Vec pos, float max_radius) = 0;
        __HOST__ __DEVICE__ virtual void update() = 0;
        __HOST__ __DEVICE__ virtual const char* getName() = 0;

    private:
        List<BoidGroup> _boidGroups; //todo hashtable

};

__HOST__ std::ostream &operator<<(std::ostream &os, NeighborStruct &ns);

#undef __HOST__
#undef __DEVICE__

#endif /* end of include guard: NEIGHBORSTRUCT_H */
