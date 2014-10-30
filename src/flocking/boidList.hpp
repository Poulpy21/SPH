
#ifndef BOIDLIST_H
#define BOIDLIST_H

#include "boid.hpp"
#include "mutex.hpp"

#ifdef __CUDACC__
#define __HOST__ __host__
#define __DEVICE__ __device__
#else
#define __HOST__
#define __DEVICE__
#endif 

struct BoidNode {
    Boid *boid;
    BoidNode *next;

    __HOST__ __DEVICE__ BoidNode(Boid *boid, BoidNode *next);
    __HOST__ __DEVICE__ ~BoidNode();
};

struct BoidList {
    BoidNode *first;
    BoidNode *last;

#ifdef __CUDACC__
    cuda::mutex readMutex;
    cuda::mutex writeMutex;
#else
    std::mutex readMutex;
    std::mutex writeMutex;
#endif

    __HOST__ __DEVICE__ BoidList();
    __HOST__ __DEVICE__ ~BoidList();

    __HOST__ __DEVICE__ void push_front(Boid *boid);
    __HOST__ __DEVICE__ void push_back(Boid *boid);
    __HOST__ __DEVICE__ void insert(Boid *boid);
    __HOST__ __DEVICE__ Boid* pop_front();
};

#undef __HOST__
#undef __DEVICE__

#endif /* end of include guard: BOIDLIST_H */
