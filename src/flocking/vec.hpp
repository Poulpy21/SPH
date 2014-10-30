
#ifndef VEC_H
#define VEC_H

#include <iostream>

#ifdef __CUDACC__
#define __HOST__ __host__
#define __DEVICE__ __device__
#else
#define __HOST__
#define __DEVICE__
#endif 

//Hybrid CPU-GPU vector structure
struct Vec {
    float x;
    float y;
    float z;

    __HOST__ __DEVICE__ Vec();
    __HOST__ __DEVICE__ Vec(const Vec &v);
    __HOST__ __DEVICE__ Vec(float x, float y, float z);
    __HOST__ __DEVICE__ ~Vec();

    __HOST__ __DEVICE__ Vec& operator= (const Vec &v);
    __HOST__ __DEVICE__ void setValue(float x, float y, float z);

    __HOST__ __DEVICE__ Vec & operator+= (const Vec &a);
    __HOST__ __DEVICE__ Vec & operator-= (const Vec &a);
    __HOST__ __DEVICE__ Vec & operator*= (const Vec &a);
    __HOST__ __DEVICE__ Vec & operator/= (const Vec &a);
    __HOST__ __DEVICE__ Vec & operator^= (const Vec &a);

    __HOST__ __DEVICE__ Vec & operator+= (float k);
    __HOST__ __DEVICE__ Vec & operator-= (float k);
    __HOST__ __DEVICE__ Vec & operator*= (float k);
    __HOST__ __DEVICE__ Vec & operator/= (float k);

    __HOST__ __DEVICE__ float normalize ();

    __HOST__ __DEVICE__ float norm () const;
    __HOST__ __DEVICE__ float squaredNorm () const;

    __HOST__ __DEVICE__ Vec unit () const;
    __HOST__ __DEVICE__ Vec orthogonalVec () const;
};
    
__HOST__ __DEVICE__ Vec operator+ (const Vec &a, const Vec &b);
__HOST__ __DEVICE__ Vec operator- (const Vec &a, const Vec &b);
__HOST__ __DEVICE__ Vec operator* (const Vec &a, const Vec &b);
__HOST__ __DEVICE__ Vec operator/ (const Vec &a, const Vec &b);
__HOST__ __DEVICE__ Vec operator^ (const Vec &a, const Vec &b);
__HOST__ __DEVICE__ float operator| (const Vec &a, const Vec &b);

__HOST__ __DEVICE__ Vec operator* (const Vec &a, double k);
__HOST__ __DEVICE__ Vec operator/ (const Vec &a, double k);

__HOST__ __DEVICE__ Vec operator* (double k, const Vec &b);
__HOST__ __DEVICE__ Vec operator/ (double k, const Vec &b);

__HOST__ __DEVICE__ bool operator!= (const Vec &a, const Vec &b);
__HOST__ __DEVICE__ bool operator== (const Vec &a, const Vec &b);

__HOST__ std::ostream & operator << (std::ostream &os, Vec &v);

#undef __HOST__
#undef __DEVICE__

#endif /* end of include guard: VEC_H */
