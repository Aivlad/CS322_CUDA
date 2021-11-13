#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "cu.cuh"

#ifndef __CUDACC__ 
#define __CUDACC__
#endif
#include <device_functions.h>

using namespace cv;
using namespace std;

__global__ void matrix_symmetry_check_shared(int* matrix, const int countLine, const int countColumn, int* vec)
{
	_/*_shared__ int cache[256];

	int iLine = blockIdx.x * blockDim.x + threadIdx.x;																	
	int iFirstElementLine = iLine * 1024;																					
	int iLastElementLine = iFirstElementLine + 511;	*/																		


	//__shared__ int cache[1024];
	//int iCache = threadIdx.x;
	//int iVec = blockIdx.x;
	//int iMatrix = blockIdx.x * blockDim.x + threadIdx.x;
	//cache[threadIdx.x] = matrix[iMatrix];
	//__syncthreads();

	//if (iCache < countColumn / 2)
	//{
	//	if (vec[iVec] == 1)
	//	{
	//		int iSymLine = 1024 - iCache - 1;
	//		vec[iVec] = cache[iCache] == cache[iSymLine] ? atomicAnd(vec + iVec, 1) : atomicAnd(vec + iVec, 0);
	//	}
	//}


}

/*
	__global__ void matrix_symmetry_check_no_atomic(int* matrix, const int countLine, const int countColumn, int* vec)
{
	int iLine = blockIdx.x * blockDim.x + threadIdx.x;																		//3
	int iFirstElementLine = iLine * 1024;																					//2
	int iLastElementLine = iFirstElementLine + 511;																			//2
	if (iFirstElementLine < countLine * countColumn && iLastElementLine < countLine * countColumn && iLine < countLine)		//7
	{
		for (int i = 0; i < 512; i++)																						//3 � ����� + 1 ���
		{
			if (vec[iLine] == 1)																							//8
			{
				vec[iLine] = matrix[iFirstElementLine + i] == matrix[iLastElementLine - i] ? 1 : 0;							//25
			}
		}																													//36 * 512 + 1 = 18433
	}
}
	
*/

void fillIntVecRes(int* vec, int n);

int sumElementsInRes(int* vec, int n);


int LaunchGPULab4(int* host_matrix, int countLine, int countColumn)
{
	int* host_is_symmetrical_lines = new int[countLine];
	fillIntVecRes(host_is_symmetrical_lines, countLine);

	int* dev_matrix;
	int* dev_is_symmetrical_lines;

	const int size_matrix = countLine * countColumn * sizeof(int);
	const int size_vector = countLine * sizeof(int);

	// ���������� ������ �������
	cudaEvent_t start, stop;
	float gpuTime = 0.0f;

	// ������� ������ � ��������� ���������� ����
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	CHECK(cudaMalloc((void**)&dev_matrix, size_matrix));											// ��������� ������ �� ����������
	CHECK(cudaMalloc((void**)&dev_is_symmetrical_lines, size_vector));								// ��������� ������ �� ����������

	CHECK(cudaMemcpy(dev_matrix, host_matrix, size_matrix, cudaMemcpyHostToDevice));								// �������� �������� �� ����������
	CHECK(cudaMemcpy(dev_is_symmetrical_lines, host_is_symmetrical_lines, size_vector, cudaMemcpyHostToDevice));	// �������� �������� �� ����������

	int countThreads = 256;																// ���������� ����� ��� ������� ����� (!!! ���-�� ����� = ���������� �������� �������)
	int countBlocks = (countLine * countColumn + countThreads - 1) / countThreads;				// ���������� ������������ ������
	printf("Blocks: %i\t Threads: %i\n", countBlocks, countThreads);

	cudaEventRecord(start, 0);                                                          // ����������� start � �������� �����

	matrix_symmetry_check_shared KERNEL_ARGS2(countBlocks, countThreads) (dev_matrix, countLine, countColumn, dev_is_symmetrical_lines);

	cudaEventRecord(stop, 0);                                                           // ����������� stop � �������� �����
	cudaEventSynchronize(stop);                                                         // ���������� ��������� ��������� ���������� ����, ��������� ����������� ������������� �� ������� stop
	cudaEventElapsedTime(&gpuTime, start, stop);                                        // ����������� ����� ����� ��������� start � stop
	printf("Time on GPU = %f milliseconds\n", gpuTime);                                 // ����� ����� ���������� � �������������
	cudaEventDestroy(start);                                                            // ������� ������� start
	cudaEventDestroy(stop);                                                             // ������� ������� stop

	CHECK(cudaGetLastError());                                                          // �������� �� ������

	CHECK(cudaMemcpy(host_is_symmetrical_lines, dev_is_symmetrical_lines, size_vector, cudaMemcpyDeviceToHost));	// �������� �������� � ����������

	CHECK(cudaFree(dev_matrix));														// �������
	CHECK(cudaFree(dev_is_symmetrical_lines));											// �������

	//printf("Memory bandwidth: %f Gb/s\n", ((size_matrix / 1024.0 / 1024.0 / 1024.0) / (gpuTime / 1000.0)));
	//printf("The matrix is symmetric ( %s )\n", host_is_symmetric ? "true" : "false");
	printf("The sum of the vector elements = %i", sumElementsInRes(host_is_symmetrical_lines, countLine));

	return 0;
}