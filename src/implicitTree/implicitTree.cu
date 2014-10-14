
#include "implicitTree.h"
#include <cassert>
#include <map>

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

__host__ unsigned int countNode(node_s *tree_h) {
    if(tree_h == NULL)
        return 0u;
    else
        return 1u + countNode(tree_h->left_child) + countNode(tree_h->right_child);
}

__host__ unsigned int buildDeviceTree(node_s *node_h, operator_node_s *node_d) {

    //check if node exists
    if(node_h == NULL)
        return 0u;

    //clone current base node struct
    cudaMemcpy(node_d,node_h,sizeof(node_s), cudaMemcpyHostToDevice);

    //clone function pointer
    if(node_h->nodeType == OPERATOR) {
        operator_node_s* dnode_h = (operator_node_s*) node_h;
        assert(operatorFunctionPointers.find(dnode_h->operatorFunc) != operatorFunctionPointers.end());
        cudaMemcpy((unsigned char*)(node_d)+sizeof(node_s), 
                (void*)operatorFunctionPointers[dnode_h->operatorFunc],
                sizeof(operatorFunction),
                cudaMemcpyHostToDevice);
    }
    else {
        density_node_s* dnode_h = (density_node_s*) node_h;
        assert(densityFunctionPointers.find(dnode_h->densityFunc) != densityFunctionPointers.end());
        cudaMemcpy((unsigned char*)(node_d)+sizeof(node_s), 
                (void*)densityFunctionPointers[dnode_h->densityFunc],
                sizeof(densityFunction),
                cudaMemcpyHostToDevice);
    }
    
    unsigned int offset = 1;
    offset += buildDeviceTree(node_h->left_child, node_d+offset);
    offset += buildDeviceTree(node_h->right_child, node_d+offset);

    return offset;
}

__host__ node_s* makeDeviceTreeFromHost(node_s *tree_h) {
   
    assert(sizeof(operator_node_s) == sizeof(density_node_s));
    const unsigned int nNode = countNode(tree_h);
    operator_node_s *data_d;
    
    cudaMalloc(&data_d, nNode*sizeof(operator_node_s));
    buildDeviceTree(tree_h, data_d);

    return (node_s*)data_d; //TODO
}

__global__ void testKernel(float *x, float *y, float *z, float *res, node_s *tree) {
    unsigned int ix = blockIdx.x;
    unsigned int iy = blockIdx.y;
    unsigned int iz = blockIdx.z;
    unsigned int i = iz*(blockDim.x*blockDim.y) + iy*blockDim.x + iz;

    res[i] = evalNode(tree,x[ix],y[iy],z[iz]);
}

__host__ void computeTestKernel(float *x, float *y, float *z, float *res, node_s *tree) {
    dim3 gridDim(10,10,10);
    dim3 blockDim(1,1,1);
    testKernel<<<gridDim,blockDim>>>(x,y,z,res,tree);
}


