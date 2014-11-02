
#ifndef SPINLOCK_H
#define SPINLOCK_H

#ifdef __CUDACC__
#define __HOST__ __host__
#define __DEVICE__ __device__
#else
#define __HOST__
#define __DEVICE__
#endif 

#include <atomic>

union hd_atomic_flag {
    std::atomic_flag cpu;
    int gpu;

    __HOST__ __DEVICE__ hd_atomic_flag();
    __HOST__ __DEVICE__ hd_atomic_flag(const hd_atomic_flag& af);
    __HOST__ __DEVICE__ hd_atomic_flag& operator= (const hd_atomic_flag& af);
    __HOST__ __DEVICE__ ~hd_atomic_flag();
};

class Spinlock {
    public: 
        __HOST__ __DEVICE__ Spinlock();
        __HOST__ __DEVICE__ Spinlock(const Spinlock& sl);
        __HOST__ __DEVICE__ Spinlock& operator= (const Spinlock& sl);
        __HOST__ __DEVICE__ ~Spinlock();

        __HOST__ __DEVICE__ void lock();
        __HOST__ __DEVICE__ void unlock();
        __HOST__ __DEVICE__ bool try_lock();
    private:
        hd_atomic_flag _flag;
};

#undef __HOST__
#undef __DEVICE__

#endif /* end of include guard: SPINLOCK_H */
