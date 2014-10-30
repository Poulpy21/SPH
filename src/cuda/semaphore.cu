
#include "semaphore.hpp"

__host__ __device__ hd_atomic_int::hd_atomic_int(int initialCount) {
#ifdef __CUDA_ARCH__
    gpu = initialCount;
#else
    cpu = initialCount;
#endif
}

__host__ __device__ hd_atomic_int::~hd_atomic_int() {
}

__host__ __device__ Semaphore::Semaphore(int initialCount) : 
    _count(initialCount) 
{}

__host__ __device__ Semaphore::~Semaphore() {
}

__host__ __device__ void Semaphore::take() {
#ifdef __CUDA_ARCH__
    int counter = _count.gpu;
    while(counter <= 0 || atomicCAS(&_count.gpu, counter, counter-1) != counter) {
        counter = _count.gpu;
    }
#else
    int counter = _count.cpu;
    while(counter <= 0 || !_count.cpu.compare_exchange_weak(counter, counter-1)) {
        counter = _count.cpu;
    }
#endif
}

__host__ __device__ void Semaphore::release() {
#ifdef __CUDA_ARCH__
    atomicAdd(&_count.gpu, 1u);
#else
    _count.cpu++;
#endif
}
