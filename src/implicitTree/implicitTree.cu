
#include "implicitTree.h"

__device__ __host__ float evalNode(node_s *tree, float x, float y, float z) {
    if(tree->nodeType == OPERATOR) {
        operator_node_s* operator_node = (operator_node_s*) tree;
        return (operator_node->operatorFunc)(
                evalNode(operator_node->node.left_child,x,y,z), 
                evalNode(operator_node->node.right_child,x,y,z)
                );
    } 
    else {
        density_node_s* density_node = (density_node_s*) tree;
        return density_node->densityFunc(x,y,z);
    }
}

__host__ node_s* makeOperatorNode(node_s* left_child, node_s* right_child, operatorFunction f) {
    operator_node_s* operator_node = (operator_node_s*) malloc(sizeof(operator_node_s));
    operator_node->node.nodeType = OPERATOR;
    operator_node->node.left_child = left_child;
    operator_node->node.right_child = right_child;
    operator_node->operatorFunc = f;
    return (node_s*) operator_node;
}

__host__ node_s* makeDensityNode(densityFunction f) {
    density_node_s* density_node = (density_node_s*) malloc(sizeof(density_node_s));
    density_node->node.nodeType = DENSITY;
    density_node->node.left_child = NULL;
    density_node->node.right_child = NULL;
    density_node->densityFunc = f;
    return (node_s*) density_node;
}

