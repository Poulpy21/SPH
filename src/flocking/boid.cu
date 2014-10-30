
#include <iostream>
#include "boid.hpp"

__device__ unsigned int boidGlobalIdGPU = 0;
#ifndef __CUDA_ARCH__          // device code
static unsigned int boidGlobalIdCPU = 0;
#endif

__host__ __device__ Boid::Boid(Vec x0, Vec v0, Vec a0, float m0) :
    id(0), m(m0), x(x0), v(v0), a(a0) {

#ifdef __CUDA_ARCH__          // device code
        id = boidGlobalIdGPU;
        boidGlobalIdGPU++;
#else                         // host code
        id = boidGlobalIdCPU;
        boidGlobalIdCPU++;
#endif
}

__host__ __device__ Boid::Boid(const Boid &b) {
    this->id = b.id;
    this->m = b.m;
    this->x = b.x;
    this->v = b.v;
    this->a = b.a;
}

__host__ __device__ Boid & Boid::operator= (const Boid &b) {
    this->id = b.id;
    this->m = b.m;
    this->x = b.x;
    this->v = b.v;
    this->a = b.a;
    return *this;
}

__host__ __device__ Boid::~Boid() {
}

__host__ std::ostream & operator << (std::ostream &os, Boid &B) {
    os << "Particule" << std::endl;
    os << "\tm = " << B.m;
    os << "\tX = " << B.x;
    os << "\tV = " << B.v;
    os << "\tA = " << B.a;
    return os;
}

