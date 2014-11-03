
#include "neighborStruct.hpp"


__host__ __device__ NeighborStruct::NeighborStruct() :
    _boidGroups()
{}

__host__ __device__ NeighborStruct::NeighborStruct(const NeighborStruct &ns) :
    _boidGroups(ns._boidGroups)
{}

__host__ __device__ NeighborStruct& NeighborStruct::operator= (const NeighborStruct& ns) {
    this->_boidGroups = ns._boidGroups;
    return *this;
}
 
__host__ __device__ NeighborStruct::~NeighborStruct() {
}

__host__ __device__ BoidGroup* NeighborStruct::getBoidGroup(const char* groupName) {
    return 0;
}

__host__ __device__ void NeighborStruct::insertBoidGroup(BoidGroup *boidGroup) {
    _boidGroups.push_back(boidGroup);
}

__host__ __device__ void NeighborStruct::removeBoidGroup(const char* groupName) {
}

__host__ std::ostream &operator<<(std::ostream &os, NeighborStruct &ns)
{
    os << ns.getName();	
    return os;
}
