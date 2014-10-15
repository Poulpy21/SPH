
#include "implicitTree.h"
#include <cassert>
#include <cstdio>
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

__host__ unsigned int countNodes(node_s *tree_h) {
    if(tree_h == NULL)
        return 0u;
    else
        return 1u + countNodes(tree_h->left_child) + countNodes(tree_h->right_child);
}

__host__ void setVal(unsigned char* host_ptr, unsigned char *val, size_t nBytes) {
        for (unsigned int i = 0; i < nBytes; i++) {
            cudaMemset(host_ptr+i, val[i], 1);
        }
}

__global__ void initNodeKernel(operator_node_s *node_d, NodeType type, node_s *leftChild, node_s *rightChild, operatorFunction function) {
    node_d->node.nodeType = type;
    node_d->node.left_child = leftChild;
    node_d->node.right_child = rightChild;
    node_d->operatorFunc = function;
}

__host__ void initNode(operator_node_s *node_d, NodeType type, node_s *leftChild, node_s *rightChild, operatorFunction function) {
    initNodeKernel<<<1,1>>>(node_d,type,leftChild,rightChild,function);
}

__host__ unsigned int buildDeviceTree(node_s *node_h, operator_node_s *node_d) {

    //check if node is NULL
    if(node_h == NULL)
        return 0;

    //create child nodes
    unsigned int offset = 1;
    operator_node_s *leftChild, *rightChild; 

    leftChild = (node_h->left_child == NULL ? NULL : node_d+offset);
    offset += buildDeviceTree(node_h->left_child, node_d+offset);
    rightChild = (node_h->right_child == NULL ? NULL : node_d+offset);
    offset += buildDeviceTree(node_h->right_child, node_d+offset);

    operatorFunction func = NULL;
    if(node_h->nodeType == OPERATOR) {
        operator_node_s* op_node_h = (operator_node_s*) node_h;
        func = operatorFunctionPointers[op_node_h->operatorFunc];
    }
    else {
        density_node_s* ds_node_h = (density_node_s*) node_h;
        func = (operatorFunction)densityFunctionPointers[ds_node_h->densityFunc];
    }
    
    printf("Creating node %p (type=%i,l=%p,r=%p,func=%p)\n", node_d, node_h->nodeType, leftChild, rightChild,func); 
    
    initNode(node_d, node_h->nodeType, (node_s*)leftChild, (node_s*)rightChild, func);

    operator_node_s testNode;
    cudaMemcpy(&testNode, node_d, sizeof(operator_node_s), cudaMemcpyDeviceToHost);
    printf("Got node \t\t(type=%i,l=%p,r=%p,func=%p)\n", testNode.node.nodeType, testNode.node.left_child, testNode.node.right_child,testNode.operatorFunc); 

    
    return offset;
}

__host__ node_s* makeDeviceTreeFromHost(node_s *tree_h) {

    assert(sizeof(operatorFunction) == sizeof(densityFunction));
    const unsigned int nNode = countNodes(tree_h);
    printf("There are %i nodes !\n", nNode);
    operator_node_s *data_d;
    
    cudaMalloc(&data_d, nNode*sizeof(operator_node_s));
    assert(buildDeviceTree(tree_h, data_d) == nNode);

    return (node_s*)data_d; 
}

__global__ void testKernel(float *x, float *y, float *z, float *res, node_s *tree) {
    unsigned int ix = blockIdx.x;
    unsigned int iy = blockIdx.y;
    unsigned int iz = blockIdx.z;
    unsigned int i = iz*(gridDim.x*gridDim.y) + iy*gridDim.x + ix;

    res[i] = evalNode(tree,x[ix],y[iy],z[iz]);
}

__host__ void computeTestKernel(float *x, float *y, float *z, float *res, node_s *tree) {
    dim3 gridDim(10,10,10);
    dim3 blockDim(1,1,1);
    testKernel<<<gridDim,blockDim>>>(x,y,z,res,tree);
}


