
#include "spinlock.hpp"


__host__ __device__ hd_atomic_flag::hd_atomic_flag() {
#ifdef __CUDA_ARCH__
    gpu = 0;
#else
    cpu.clear();
#endif
}

__host__ __device__ hd_atomic_flag::~hd_atomic_flag() {
#ifdef __CUDA_ARCH__
    gpu = 0;
#else
    cpu.clear();
#endif
}

__host__ __device__ Spinlock::Spinlock() : _flag() {
}

__host__ __device__ Spinlock::~Spinlock() {
}

__host__ __device__ void Spinlock::lock() {
#ifdef __CUDA_ARCH__
    while(atomicCAS(&_flag.gpu, 0, 1) != 0);
#else
    while(_flag.cpu.test_and_set(std::memory_order_acquire));
#endif
}

__host__ __device__ bool Spinlock::try_lock() {
#ifdef __CUDA_ARCH__
    return atomicCAS(&_flag.gpu, 0, 1) == 0;
#else
    return _flag.cpu.test_and_set(std::memory_order_acquire);
#endif
}

__host__ __device__ void Spinlock::unlock() {
#ifdef __CUDA_ARCH__
    atomicExch(&_flag.gpu, 1);
#else
    _flag.cpu.clear(std::memory_order_release); 
#endif
}


