
#ifndef SERIALUNIFORMGRID_H
#define SERIALUNIFORMGRID_H

#include "neighborStruct.hpp"

struct BoidGroup {
    Boid *boids;
    std::string groupName;
    unsigned int nBoids;
}

class SerialUniformGrid : public NeighborStruct {
  
public:
    SerialUniformGrid();
    ~SerialUniformGrid();
    
    void insertBoids(std::string groupName, Boid** boids, unsigned int nBoids);
    
    Boid* getBoidGroup(std::string groupName);
    std::list<Boid*> getNearbyNeighbors(std::string groupName, Vec pos, float max_radius);

	void update();

	const std::string getName();

private:
    std::vector<std::vector<std::map<std::string, std::list<Boid*>>>> grid; 

#endif /* end of include guard: SERIALUNIFORMGRID_H */
