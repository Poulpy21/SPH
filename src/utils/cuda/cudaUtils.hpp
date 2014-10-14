
#ifndef _CUDA_UTILS_
#define _CUDA_UTILS_

#include <iostream>
#include "cuda.h"
#include "cuda_runtime.h"

#define CHECK_CUDA_ERRORS(ans) { gpuAssert((ans), __FILE__, __LINE__); }
void gpuAssert(cudaError_t code, const std::string &file, int line, bool abort = true);
void checkKernelExecution();

#ifndef __CUDACC__

#include <log4cpp/Category.hh>
#include "log.hpp"

using namespace log4cpp;

class CudaUtils {

	public:
		static void printCudaDevices(std::ostream &outputStream);
		static void logCudaDevices(log4cpp::Category &log_output);
		
};

#endif

#endif
