
#ifndef IMPLICITTREE_H
#define IMPLICITTREE_H

#include "implicitFunc.h"

#ifdef __CUDACC__
#define __HOST__ __host__
#define __DEVICE__ __device__
#else
#define __HOST__
#define __DEVICE__
#endif

extern "C" {

    enum NodeType {
        OPERATOR,
        DENSITY
    };

    typedef struct __align__(8) node {
        NodeType nodeType;
        struct node *left_child;
        struct node *right_child;
    } node_s;

    typedef struct __align__(8) operator_node {
        struct node node;
        operatorFunction operatorFunc;
    } operator_node_s;
    
    typedef struct __align__(8) density_node {
        struct node node;
        densityFunction densityFunc;
    } density_node_s;

    typedef node_s* implicitTree;

    __HOST__ node_s* makeDensityNode(densityFunction f);
    __HOST__ node_s* makeOperatorNode(node_s* left_child, node_s* right_child, operatorFunction f);
   
    __HOST__ __DEVICE__ float evalNode(node_s *tree, float x, float y, float z);
}

#undef __HOST__
#undef __DEVICE__

#endif /* end of include guard: IMPLICITTREE_H */
