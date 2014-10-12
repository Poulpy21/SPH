
#ifndef _CUDA_UTILS_
#define _CUDA_UTILS_

#include <iostream>
#include <log4cpp/Category.hh>
#include "cuda.h"
#include "cuda_runtime.h"

#include "log.hpp"

using namespace log4cpp;

#define CHECK_CUDA_ERRORS(ans) { gpuAssert((ans), __FILE__, __LINE__); }

class CudaUtils {

	public:
		static void printCudaDevices(std::ostream &outputStream);
		static void logCudaDevices(log4cpp::Category &log_output);
		
};

void gpuAssert(cudaError_t code, const std::string &file, int line, bool abort = true);
void checkKernelExecution();

#endif
