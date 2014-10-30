
#ifndef BOIDGROUP_H
#define BOIDGROUP_H

struct BoidGroup {
    std::string groupName;
    unsigned int nBoids;
    Boid *boids;
    
    BoidGroup(std::string groupName, Boid *boids, unsigned int nBoids) :
        groupName(groupName), nBoids(nBoids), boids(boids) { }
};

#endif /* end of include guard: BOIDGROUP_H */
