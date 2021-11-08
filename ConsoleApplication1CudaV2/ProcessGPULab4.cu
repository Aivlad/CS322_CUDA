#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "cu.cuh"

#ifndef __CUDACC__ 
#define __CUDACC__
#endif
#include <device_functions.h>

using namespace cv;
using namespace std;

const int portion = 1024;

__global__ void matrix_symmetry_check_shared(int* matrix, const int countLine, const int countColumn, bool* isSymmetric)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i >= countLine * countColumn)
		return;

	__shared__ int cache[portion];

	cache[threadIdx.x] = matrix[i];
	
	__syncthreads();

	*isSymmetric = cache[threadIdx.x] == cache[portion - 1 - threadIdx.x] ? true : false;
}

int LaunchGPULab4(int* host_matrix, int countLine, int countColumn)
{
	bool host_is_symmetric = true;	// ������������� �������, ��� ������� ������������

	int* dev_matrix;
	bool* dev_is_symmetric;

	const int size = countLine * countColumn * sizeof(int);

	// ���������� ������ �������
	cudaEvent_t start, stop;
	float gpuTime = 0.0f;

	// ������� ������ � ��������� ���������� ����
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	CHECK(cudaMalloc((void**)&dev_matrix, size));											// ��������� ������ �� ����������
	CHECK(cudaMalloc((void**)&dev_is_symmetric, sizeof(bool)));                             // ��������� ������ �� ����������

	CHECK(cudaMemcpy(dev_matrix, host_matrix, size, cudaMemcpyHostToDevice));						// �������� �������� �� ����������
	CHECK(cudaMemcpy(dev_is_symmetric, &host_is_symmetric, sizeof(bool), cudaMemcpyHostToDevice));	// �������� �������� �� ����������

	int countThreads = countColumn;																// ���������� ����� ��� ������� ����� (!!! ���-�� ����� = ���������� �������� �������)
	int countBlocks = (countLine * countColumn + countThreads - 1) / countThreads;				// ���������� ������������ ������
	printf("Blocks: %i\t Threads: %i\n", countBlocks, countThreads);

	cudaEventRecord(start, 0);                                                          // ����������� start � �������� �����

	matrix_symmetry_check_shared KERNEL_ARGS2(countBlocks, countThreads) (dev_matrix, countLine, countColumn, dev_is_symmetric);

	cudaEventRecord(stop, 0);                                                           // ����������� stop � �������� �����
	cudaEventSynchronize(stop);                                                         // ���������� ��������� ��������� ���������� ����, ��������� ����������� ������������� �� ������� stop
	cudaEventElapsedTime(&gpuTime, start, stop);                                        // ����������� ����� ����� ��������� start � stop
	printf("Time on GPU = %f milliseconds\n", gpuTime);                                 // ����� ����� ���������� � �������������
	cudaEventDestroy(start);                                                            // ������� ������� start
	cudaEventDestroy(stop);                                                             // ������� ������� stop

	CHECK(cudaGetLastError());                                                          // �������� �� ������

	CHECK(cudaMemcpy(&host_is_symmetric, dev_is_symmetric, sizeof(bool), cudaMemcpyDeviceToHost));	// �������� �������� � ����������

	CHECK(cudaFree(dev_matrix));														// �������
	CHECK(cudaFree(dev_is_symmetric));													// �������

	printf("Memory bandwidth: %f Gb/s\n", ((size / 1024.0 / 1024.0 / 1024.0) / (gpuTime / 1000.0)));
	printf("The matrix is symmetric ( %s )\n", host_is_symmetric ? "true" : "false");

	return 0;
}