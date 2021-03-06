
#include <iostream>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#pragma region ?????????? ?? ??????????

// ??????????? ?????????? ????
int getSPcores(cudaDeviceProp devProp)
{
    int cores = 0;
    int mp = devProp.multiProcessorCount;
    switch (devProp.major)
    {
    case 2: // Fermi
        if (devProp.minor == 1) cores = mp * 48;
        else cores = mp * 32;
        break;
    case 3: // Kepler
        cores = mp * 192;
        break;
    case 5: // Maxwell
        cores = mp * 128;
        break;
    case 6: // Pascal
        if (devProp.minor == 1) cores = mp * 128;
        else if (devProp.minor == 0) cores = mp * 64;
        else printf("Unknown device type\n");
        break;
    case 7: // Volta
        if (devProp.minor == 0) cores = mp * 64;
        else printf("Unknown device type\n");
        break;
    default:
        printf("Unknown device type\n");
        break;
    }
    return cores;
}

// ????? ?????????? ?? ????????? ??????????
void print_cuda_device_info(cudaDeviceProp& prop)
{
    printf("Device name:                                        %s\n", prop.name);
    printf("Global memory available on device:                  %zu (%zu MByte)\n", prop.totalGlobalMem, prop.totalGlobalMem / 1024 / 1024);
    printf("Shared memory available per block:                  %zu\n", prop.sharedMemPerBlock);
    printf("Count of 32-bit registers available per block:      %i\n", prop.regsPerBlock);
    printf("Warp size in threads:                               %i\n", prop.warpSize);
    printf("Maximum pitch in bytes allowed by memory copies:    %zu\n", prop.memPitch);
    printf("Maximum number of threads per block:                %i\n", prop.maxThreadsPerBlock);
    printf("Maximum size of each dimension of a block[0]:       %i\n", prop.maxThreadsDim[0]);
    printf("Maximum size of each dimension of a block[1]:       %i\n", prop.maxThreadsDim[1]);
    printf("Maximum size of each dimension of a block[2]:       %i\n", prop.maxThreadsDim[2]);
    printf("Maximum size of each dimension of a grid[0]:        %i\n", prop.maxGridSize[0]);
    printf("Maximum size of each dimension of a grid[1]:        %i\n", prop.maxGridSize[1]);
    printf("Maximum size of each dimension of a grid[2]:        %i\n", prop.maxGridSize[2]);
    printf("Clock frequency in kilohertz:                       %i\n", prop.clockRate);
    printf("totalConstMem:                                      %zu\n", prop.totalConstMem);
    printf("Major compute capability:                           %i\n", prop.major);
    printf("Minor compute capability:                           %i\n", prop.minor);
    printf("Number of multiprocessors on device:                %i\n", prop.multiProcessorCount);
    printf("Count of cores:                                     %i\n", getSPcores(prop));
    printf("\n\n\n");
}

#pragma endregion



