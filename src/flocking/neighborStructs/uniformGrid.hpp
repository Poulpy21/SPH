
#ifndef UNIFORMGRID_H
#define UNIFORMGRID_H

#include "boidGroup.hpp"
#include "neighborStruct.hpp"
#include "simplyLinkedList.hpp"

#ifdef __CUDACC__
#define __HOST__ __host__
#define __DEVICE__ __device__
#else
#define __HOST__
#define __DEVICE__
#endif 

class UniformGrid : public NeighborStruct {
  
public:
    __HOST__ __DEVICE__ UniformGrid(float minRadius, unsigned int staticArraySize);
    __HOST__ __DEVICE__ ~UniformGrid();
    
    __HOST__ __DEVICE__ void insertBoids(BoidGroup &group);
    
    __HOST__ __DEVICE__ Boid** getBoidGroup(const char* groupName);
    
    __HOST__ __DEVICE__ List<Boid> getNearbyNeighbors(const char* groupName, Vec pos, float max_radius);

	__HOST__ __DEVICE__ void update();

	__HOST__ __DEVICE__ const char* getName();

private:
    List<Boid> **_grid;

};

#undef __HOST__
#undef __DEVICE__

#endif /* end of include guard: UNIFORMGRID_H */
