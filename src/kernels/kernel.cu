
#include "cuda.h"
#include "cuda_runtime.h"

__global__ void moveVertex(float *vertex,
		float dx) {
	
	float pi = 3.14;
	float dtheta = pi/50.0f;

	float y = vertex[3*blockIdx.x + 1];
	float z = vertex[3*blockIdx.x + 2];
	float Ct = cos(dtheta);
	float St = sin(dtheta);

	switch(threadIdx.x) {
		case 0: 
			vertex[3*blockIdx.x + threadIdx.x] += dx;
			break;
		case 1:
			vertex[3*blockIdx.x + threadIdx.x] = y*Ct+z*St;
			break;
		case 2:
			vertex[3*blockIdx.x + threadIdx.x] = -y*St+z*Ct;
			break;
	}
}

