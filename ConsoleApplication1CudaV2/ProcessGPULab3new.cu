#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "cu.cuh"

using namespace cv;
using namespace std;


__global__ void matrix_symmetry_check_no_atomic(int* matrix, const int countLine, const int countColumn, int* vec)
{
	int iLine = blockIdx.x * blockDim.x + threadIdx.x;																		//3
	int iFirstElementLine = iLine * 1024;																					//2
	int iLastElementLine = iFirstElementLine + 511;																			//2
	if (iFirstElementLine < countLine * countColumn && iLastElementLine < countLine * countColumn && iLine < countLine)		//7
	{
		for (int i = 0; i < 512; i++)																						//3 в цикле + 1 вне
		{
			if (vec[iLine] == 1)																							//8
			{
				vec[iLine] = matrix[iFirstElementLine + i] == matrix[iLastElementLine - i] ? 1 : 0;							//25
			}
		}																													//36 * 512 + 1 = 18433
	}
}


void fillIntVecRes(int* vec, int n);

int sumElementsInRes(int* vec, int n);

int LaunchGPULab3new(int* host_matrix, int countLine, int countColumn)
{
	int* host_is_symmetrical_lines = new int[countLine];
	fillIntVecRes(host_is_symmetrical_lines, countLine);

	int* dev_matrix;
	int* dev_is_symmetrical_lines;

	const int size_matrix = countLine * countColumn * sizeof(int);
	const int size_vector = countLine * sizeof(int);

	// Переменные замера времени
	cudaEvent_t start, stop;
	float gpuTime = 0.0f;

	// События начала и окончания выполнения ядра
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	CHECK(cudaMalloc((void**)&dev_matrix, size_matrix));											// выделение памяти на устройстве
	CHECK(cudaMalloc((void**)&dev_is_symmetrical_lines, size_vector));								// выделение памяти на устройстве

	CHECK(cudaMemcpy(dev_matrix, host_matrix, size_matrix, cudaMemcpyHostToDevice));								// копируем значение на устройство
	CHECK(cudaMemcpy(dev_is_symmetrical_lines, host_is_symmetrical_lines, size_vector, cudaMemcpyHostToDevice));	// копируем значение на устройство

	int countThreads = countColumn;																
	int countBlocks = (countLine + countThreads - 1) / countThreads;				// количество параллельных блоков
	printf("Blocks: %i\t Threads: %i\n", countBlocks, countThreads);

	cudaEventRecord(start, 0);                                                          // привязываем start к текущему месту

	matrix_symmetry_check_no_atomic KERNEL_ARGS2(countBlocks, countThreads) (dev_matrix, countLine, countColumn, dev_is_symmetrical_lines);

	cudaEventRecord(stop, 0);                                                           // привязываем stop к текущему месту
	cudaEventSynchronize(stop);                                                         // дожидаемся реального окончания выполнения ядра, используя возможность стнхронизации по событию stop
	cudaEventElapsedTime(&gpuTime, start, stop);                                        // запрашиваем время между событиями start и stop
	printf("Time on GPU = %f milliseconds\n", gpuTime);                                 // здесь сразу вычисление в миллисекундах
	cudaEventDestroy(start);                                                            // убиваем событие start
	cudaEventDestroy(stop);                                                             // убиваем событие stop

	CHECK(cudaGetLastError());                                                          // проверка на ошибки

	CHECK(cudaMemcpy(host_is_symmetrical_lines, dev_is_symmetrical_lines, size_vector, cudaMemcpyDeviceToHost));	// копируем значение с устройства

	CHECK(cudaFree(dev_matrix));														// очистка
	CHECK(cudaFree(dev_is_symmetrical_lines));											// очистка

	//printf("Memory bandwidth: %f Gb/s\n", ((size_matrix / 1024.0 / 1024.0 / 1024.0) / (gpuTime / 1000.0)));
	//printf("The matrix is symmetric ( %s )\n", host_is_symmetric ? "true" : "false");
	printf("The sum of the vector elements = %i", sumElementsInRes(host_is_symmetrical_lines, countLine));

	return 0;
}