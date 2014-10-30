
#include "boidList.hpp"


__host__ __device__ BoidNode::BoidNode(Boid *boid, BoidNode *next) :
    boid(boid), next(next) {}

    __host__ __device__ BoidNode::~BoidNode() {}

    __host__ __device__ BoidList::BoidList() : first(0), last(0) {}
    __host__ __device__ BoidList::~BoidList() {}

    __host__ __device__ void BoidList::push_front(Boid *boid) {
        if(first == 0 && last == 0) {
            BoidNode *bn = new BoidNode(boid, 0);
            first = bn;
            last = bn;
        }
        else {
            BoidNode *bn = new BoidNode(boid, first);
            first = bn;
        }
    }

__host__ __device__ Boid* BoidList::pop_front() {
    if(first == 0)
        return 0;

    Boid *boidBuffer = first->boid;
    BoidNode *nextNode = first->next;

    delete first;

    last = (first == last ? 0 : last);
    first = nextNode;

    return boidBuffer;
}

__host__ __device__ void BoidList::push_back(Boid *boid) {
    if(first == 0 && last == 0) {
        BoidNode *bn = new BoidNode(boid, 0);
        first = bn;
        last = bn;
    }
    else {
        BoidNode *bn = new BoidNode(boid, 0);
        last->next = bn;
        last = bn;
    }
}

