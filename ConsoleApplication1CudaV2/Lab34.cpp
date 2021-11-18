#include <cmath>
#include <iostream>
#include <fstream>

#include "cu.cuh"

#define N 50000
#define M 1024

#pragma region Генерация матрицы
// случайная генерация матрицы
void generateMatrix(int* matrix, int size)
{
	srand(time(NULL));
	for (size_t i = 0; i < size; i++)
	{
		matrix[i] = rand() % 100;
	}
}

// случайная генерация симметричной матрицы
void generateMatrix(int* matrix, int countLine, int countColumn)
{
	srand(time(NULL));
	for (int i = 0; i < countLine; i++)
	{
		for (int j = 0; j < countColumn / 2; j++)
		{
			int num = i;
			//int num = rand() % 100;
			matrix[i * countColumn + j] = num;
			matrix[(i + 1) * countColumn - 1 - j] = num;
		}
	}
}

// создание копии матрицы
void copyMatrix(int* source, int* recipient, int size)
{
	for (size_t i = 0; i < size; i++)
	{
		recipient[i] = source[i];
	}
}
#pragma endregion

#pragma region Вывод матрицы
// вывод матрицы на печать в консоль (не рекомедуется т.к. матрица большая)
void printMatrix(int* matrix, int countLine, int countColumn)
{
	for (int i = 0; i < countLine; i++)
	{
		for (int j = 0; j < countColumn; j++)
		{
			printf("%i\t", matrix[i * countColumn + j]);
		}
		printf("\n");
	}
	printf("\n");
}

// вывод матрицы на печать в файл (не рекомедуется т.к. матрица большая)
void printFilleMatrix(int* matrix, int countLine, int countColumn)
{
	std::ofstream outf("matrix.txt");
	if (!outf)
	{
		std::cout << "File could not be opened for writing!" << std::endl;
		exit(1);
	}

	for (int i = 0; i < countLine; i++)
	{
		for (int j = 0; j < countColumn; j++)
		{
			outf << matrix[i * countColumn + j] << " ";
		}
		outf << std::endl;
	}
	outf << std::endl;
}
#pragma endregion

void fillIntVecRes(int* vec, int n)
{
	for (int i = 0; i < n; i++) {
		vec[i] = 1;
	}
}

int sumElementsInRes(int* vec, int n) {
	int sum = 0;
	for (int i = 0; i < n; i++) {
		sum += vec[i];
		/*if (vec[i] == 0) {
			printf("------>%i not symmetrical\n" , i);
		}*/
	}
	return sum;
}

int LaunchGPULab3(int* matrix, int countLine, int countColumn);
int LaunchGPULab3new(int* matrix, int countLine, int countColumn);
int LaunchGPULab4(int* matrix, int countLine, int countColumn);

int Launch()
{
	// Вывод информации об устрйосве
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);  // извлечение информации о нужном устройстве
	print_cuda_device_info(prop);       // вывод информации об извлеченном устрйостве

	// Работа с матрицей
	printf("Creation of a %i-by-%i matrix has begun\n", N, M);
	int* source = new int[N * M];		// создание
	//generateMatrix(source, N * M);		// заполнение просто случайно (шанс получения симметрии очень низок)
	generateMatrix(source, N, M);		// заполнение симметрично-случайно
	printf("Generation completed\n\n");
	int s = 0;
	for (int i = 0; i < N; i++) {
		if (rand() % 2 == 0) {
			source[i * 1024 + rand() % M] = -1;
			s++;
		}
	}
	std::cout << "True sym lines count = " << N-s << "\n";

	// NVIDIA GeForce 940MX (type DDR3)
	// Теоретическая пиковая пропускная способность (bandwidth): 16.02 GB/s ((c) Википедия)
	//printf("NVIDIA GeForce 940MX (type DDR3) has theoretical peak throughput (bandwidth): 16.02 GB/s\n");

	printf("\nLab3 with atomic:\n");
	LaunchGPULab3(source, N, M);

	printf("\n\nLab3 without atomic:\n");
	LaunchGPULab3new(source, N, M);

	//printf("\nLab4:\n");
	//LaunchGPULab4(source, N, M);


	return 0;
}