#include <cmath>
#include <iostream>
#include <fstream>

#include "cu.cuh"

#define N 50000
#define M 1024

#pragma region ��������� �������
// ��������� ��������� �������
void generateMatrix(int* matrix, int size)
{
	srand(time(NULL));
	for (size_t i = 0; i < size; i++)
	{
		matrix[i] = rand() % 100;
	}
}

// ��������� ��������� ������������ �������
void generateMatrix(int* matrix, int countLine, int countColumn)
{
	srand(time(NULL));
	for (int i = 0; i < countLine; i++)
	{
		for (int j = 0; j < countColumn / 2; j++)
		{
			int num = 1;
			//int num = rand() % 100;
			matrix[i * countColumn + j] = num;
			matrix[(i + 1) * countColumn - 1 - j] = num;
		}
	}
}

// �������� ����� �������
void copyMatrix(int* source, int* recipient, int size)
{
	for (size_t i = 0; i < size; i++)
	{
		recipient[i] = source[i];
	}
}
#pragma endregion

#pragma region ����� �������
// ����� ������� �� ������ � ������� (�� ������������ �.�. ������� �������)
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

// ����� ������� �� ������ � ���� (�� ������������ �.�. ������� �������)
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


int LaunchGPULab3(int* matrix, int countLine, int countColumn);
//int LaunchGPULab4(int* matrix, int countLine, int countColumn);

int Launch()
{
	// ����� ���������� �� ���������
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);  // ���������� ���������� � ������ ����������
	print_cuda_device_info(prop);       // ����� ���������� �� ����������� ����������

	// ������ � ��������
	printf("Creation of a %i-by-%i matrix has begun\n", N, M);
	int* source = new int[N * M];		// ��������
	//generateMatrix(source, N * M);		// ���������� ������ �������� (���� ��������� ��������� ����� �����)
	generateMatrix(source, N, M);		// ���������� �����������-��������
	printf("Generation completed\n\n");
	source[0] = 2;
	source[2050] = 2;
	source[51199999] = 2;


	// NVIDIA GeForce 940MX (type DDR3)
	// ������������� ������� ���������� ����������� (bandwidth): 16.02 GB/s ((c) ���������)
	//printf("NVIDIA GeForce 940MX (type DDR3) has theoretical peak throughput (bandwidth): 16.02 GB/s\n");

	printf("\nLab3:\n");
	LaunchGPULab3(source, N, M);

	//printf("\nLab4:\n");
	//LaunchGPULab4(source, N, M);


	return 0;
}