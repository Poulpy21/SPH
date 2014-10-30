
#ifdef __CUDACC__
#ifndef MUTEX_H
#define MUTEX_H

namespace cuda {

    class mutex {
        public: 
            __device__ mutex();
            __device__ ~mutex();

            __device__ void lock();
            __device__ void unlock();
            __device__ bool try_lock();
            __device__ bool try_lock(unsigned int nTries);
        private:
            int _mutex;
    };

};

#endif /* end of include guard: MUTEX_H */
#endif /* end of ifdef __CUDACC__ */
