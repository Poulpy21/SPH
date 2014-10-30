
#ifndef SEMAPHORE_H
#define SEMAPHORE_H

#ifdef __CUDACC__
#define __HOST__ __host__
#define __DEVICE__ __device__
#else
#define __HOST__
#define __DEVICE__
#endif 

#include <atomic>

// CECI N'EST QU'UNE ILLUSION //
union hd_atomic_int {
    std::atomic<int> cpu;
    int gpu;

    __HOST__ __DEVICE__ hd_atomic_int(int initialCount);
    __HOST__ __DEVICE__ ~hd_atomic_int();
};
// // // // // // // // // //

class Semaphore {
    public: 
        __HOST__ __DEVICE__ Semaphore(int initialCount);
        __HOST__ __DEVICE__ ~Semaphore();

        __HOST__ __DEVICE__ void take();
        __HOST__ __DEVICE__ void release();

    private:
        hd_atomic_int _count;
};

#undef __HOST__
#undef __DEVICE__

#endif /* end of include guard: SEMAPHORE_H */
