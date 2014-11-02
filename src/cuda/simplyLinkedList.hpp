
#ifndef SIMPLYLINKEDLIST_H
#define SIMPLYLINKEDLIST_H

#include <pthread.h>
#include "spinlock.hpp"

#ifdef __CUDACC__
#define __HOST__ __host__
#define __DEVICE__ __device__
#else
#define __HOST__
#define __DEVICE__
#endif 

template <typename T>
struct Node {
    T *data;
    Node<T> *next;

    __HOST__ __DEVICE__ Node();
    __HOST__ __DEVICE__ Node(T *data, Node *next);
    __HOST__ __DEVICE__ Node(const Node<T> &node);
    __HOST__ __DEVICE__ Node<T>& operator= (const Node<T> &node);
    __HOST__ __DEVICE__ ~Node();
};

template <typename T>
struct List {
    Node<T> *first;
    Node<T> *last;

    Spinlock readMutex;
    Spinlock writeMutex;
    unsigned int nReaders;

    __HOST__ __DEVICE__ List();
    __HOST__ __DEVICE__ List(const List<T>& list);
    __HOST__ __DEVICE__ List<T>& operator= (const List<T>& list);
    __HOST__ __DEVICE__ ~List();

    __HOST__ __DEVICE__ void push_front(T *data);
    __HOST__ __DEVICE__ void push_back(T *data);
    __HOST__ __DEVICE__ void insert(T *data);
    __HOST__ __DEVICE__ T* pop_front();
};

#undef __HOST__
#undef __DEVICE__

#endif /* end of include guard: SIMPLYLINKEDLIST_H */
