
//le flag debug est passé avec make debug mais on le met
//par defaut pour le moment
#ifndef _DEBUG_LEVEL 
#define _DEBUG_LEVEL 0
#endif

//tout passe en DEBUG avec make debug
#ifndef _CONSOLE_LOG_LEVEL
#define _CONSOLE_LOG_LEVEL INFO
#endif

#ifndef _FILE_LOG_LEVEL
#define _FILE_LOG_LEVEL DEBUG
#endif

#ifdef _DEBUG
#define CHECK_CUDA_ERRORS(ans) { gpuAssert((ans), __FILE__, __LINE__); }
#define CHECK_KERNEL_EXECUTION() { checkKernelExecution(); }
#define CHECK_OPENGL_ERRORS() { utils::glAssert( __FILE__, __LINE__); }
#define PRINTD(...) \
do {\
        printf(__VA_ARGS__);\
} while(0)
#else
#define CHECK_CUDA_ERRORS(ans) ans
#define CHECK_KERNEL_EXECUTION()
#define CHECK_OPENGL_ERRORS() 
#define PRINTD(...)
#endif

#if defined(__CUDACC__) // NVCC
   #define ALIGN(n) __align__(n)
#elif defined(__GNUC__) // GCC
  #define ALIGN(n) __attribute__((aligned(n)))
#elif defined(_MSC_VER) // MSVC
  #define ALIGN(n) __declspec(align(n))
#else
  #error "Please provide a definition for ALIGN macro for your host compiler (in utils/defines.hpp) !"
#endif

