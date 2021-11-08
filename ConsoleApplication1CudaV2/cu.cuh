#ifndef CU_CUH
#define CU_CUH

#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#pragma region Макрос для Intellisense
#ifdef __CUDACC__
#define KERNEL_ARGS2(grid, block) <<< grid, block >>>
#define KERNEL_ARGS3(grid, block, sh_mem) <<< grid, block, sh_mem >>>
#define KERNEL_ARGS4(grid, block, sh_mem, stream) <<< grid, block, sh_mem, stream >>>
#else
#define KERNEL_ARGS2(grid, block)
#define KERNEL_ARGS3(grid, block, sh_mem)
#define KERNEL_ARGS4(grid, block, sh_mem, stream)
#endif
#pragma endregion


void print_cuda_device_info(cudaDeviceProp& prop);

static void Check(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        printf("Error: %s at line %d in file %s\n", cudaGetErrorString(err), line, file);
        exit(EXIT_FAILURE);
    }
}
#define CHECK( err ) (Check( err, __FILE__, __LINE__ ))

#endif