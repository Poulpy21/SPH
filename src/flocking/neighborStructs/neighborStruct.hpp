
#ifndef NEIGHBORSTRUCT_H
#define NEIGHBORSTRUCT_H

#include <list>
#include "boid.hpp"
#include "boidGroup.hpp"

class NeighborStruct {
    public:
        virtual ~NeighborStruct() {}

        void insertBoidGroup(BoidGroup *boidGroup);
        void removeBoidGroup(std::string groupName);

        BoidGroup* getBoidGroup(std::string groupName);

        virtual std::list<Boid*> getNearbyNeighbors(std::string groupName, Vec pos, float max_radius) = 0;

        virtual void update() = 0;

        virtual const std::string getName() = 0;

    protected:
        NeighborStruct() {}
    private:
        std::map<std::string, BoidGroup*> _boidGroups; 
};

ostream &operator<<(ostream &os, NeighborStruct &ns)
{
    os << ns.getName();	
    return os;
}


#endif /* end of include guard: NEIGHBORSTRUCT_H */
