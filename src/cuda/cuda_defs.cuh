#ifndef CUDA_DEFS_CUH
#define CUDA_DEFS_CUH

#ifndef __CUDACC__
#define __host__
#define __device__
#endif
#define ANY __host__ __device__
#define CPU __host__
#define GPU __device__

#define CUDA_CHECK(call)                                                                                               \
    {                                                                                                                  \
        const cudaError_t error = call;                                                                                \
        if (error != cudaSuccess) {                                                                                    \
            std::cerr << "CUDA Error: " << cudaGetErrorString(error) << " at " << __FILE__ << ":" << __LINE__          \
                      << std::endl;                                                                                    \
            exit(1);                                                                                                   \
        }                                                                                                              \
    }

#endif
