#pragma once

// Compile with -DUSE_MUSA for Moore Threads MUSA; otherwise use CUDA.
// Keeping vendor names in this one file leaves all numerical kernels portable.
#if defined(USE_MUSA)
#include <musa_runtime.h>
#include <murand_kernel.h>

using gpuError_t = musaError_t;
using gpuRngState = murandStatePhilox4_32_10_t;

#define GPU_SUCCESS musaSuccess
#define gpuGetErrorString musaGetErrorString
#define gpuMalloc musaMalloc
#define gpuFree musaFree
#define gpuMemcpy musaMemcpy
#define gpuMemcpyHostToDevice musaMemcpyHostToDevice
#define gpuMemcpyDeviceToHost musaMemcpyDeviceToHost
#define gpuGetLastError musaGetLastError
#define gpuDeviceSynchronize musaDeviceSynchronize
#define gpuRngInit murand_init
#define gpuRngNext murand
#define GPU_BACKEND_NAME "MUSA"
#else
#include <cuda_runtime.h>
#include <curand_kernel.h>

using gpuError_t = cudaError_t;
using gpuRngState = curandStatePhilox4_32_10_t;

#define GPU_SUCCESS cudaSuccess
#define gpuGetErrorString cudaGetErrorString
#define gpuMalloc cudaMalloc
#define gpuFree cudaFree
#define gpuMemcpy cudaMemcpy
#define gpuMemcpyHostToDevice cudaMemcpyHostToDevice
#define gpuMemcpyDeviceToHost cudaMemcpyDeviceToHost
#define gpuGetLastError cudaGetLastError
#define gpuDeviceSynchronize cudaDeviceSynchronize
#define gpuRngInit curand_init
#define gpuRngNext curand
#define GPU_BACKEND_NAME "CUDA"
#endif

