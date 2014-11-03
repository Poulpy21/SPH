

#include "boidGroup.hpp"

const char* groupName;
unsigned int nBoids;
Boid *boids;


__host__ __device__ BoidGroup::BoidGroup() :
    groupName(0), nBoids(0), boids(0) {
    }

__host__ __device__ BoidGroup::BoidGroup(const char* groupName, Boid *boids, unsigned int nBoids) :
    groupName(groupName), nBoids(nBoids), boids(boids) {
    }

__host__ __device__ BoidGroup::BoidGroup(const BoidGroup &bg) :
    groupName(bg.groupName), nBoids(bg.nBoids), boids(bg.boids) {
    }

__host__ __device__ BoidGroup::~BoidGroup() {
}


__host__ __device__ BoidGroup& BoidGroup::operator= (const BoidGroup &bg) {
    this->groupName = bg.groupName;
    this->nBoids = bg.nBoids;
    this->boids = bg.boids;
    return *this;
}


__host__ __device__ unsigned long long int hashName() {
    //TODO
    return 0;
}

