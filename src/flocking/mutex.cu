
#include "mutex.hpp"

namespace cuda {

    __device__ mutex::mutex() : _mutex(0) {
    }

    __device__ mutex::~mutex() {
    }

    __device__ void mutex::lock() {
        while(atomicCAS(&_mutex, 0, 1) != 0);
    }

    __device__ bool mutex::try_lock() {
        return atomicCAS(&_mutex, 0, 1) == 0;
    }
    
    __device__ bool mutex::try_lock(unsigned int nTries) {
        for (unsigned int i = 0; i < nTries; i++) {
            if (atomicCAS(&_mutex, 0, 1) == 0)
                return true;
        }
        return false;
    }

    __device__ void mutex::unlock() {
        atomicExch(&_mutex, 1);
    }

};

