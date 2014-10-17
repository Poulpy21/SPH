
//le flag debug est passé avec make debug mais on le met
//par defaut pour le moment
#ifndef __DEBUG
#define __DEBUG
#endif

#ifndef __DEBUG_LEVEL 
#define __DEBUG_LEVEL 1
#endif

#define PRINTD(...) \
do {\
    if(__DEBUG_LEVEL > 0)\
        printf(__VA_ARGS__);\
} while(0)

//tout passe en DEBUG avec make debug
#ifndef __CONSOLE_LOG_LEVEL
#define __CONSOLE_LOG_LEVEL INFO
#endif

#ifndef __FILE_LOG_LEVEL
#define __FILE_LOG_LEVEL DEBUG
#endif

#define CHECK_CUDA_ERRORS(ans) { gpuAssert((ans), __FILE__, __LINE__); }
#define CHECK_KERNEL_EXECUTION() { checkKernelExecution(); }
#define CHECK_OPENGL_ERRORS() { utils::glAssert( __FILE__, __LINE__); }
