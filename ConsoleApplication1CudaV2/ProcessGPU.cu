
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "cu.cuh"

using namespace cv;
using namespace std;

// GPU: ���������� � ��������� ������ ����������� �������� "�������"
// threadIdx - ���������� ������ � ����� �������
// blockIdx - ���������� ����� ������� � �����
// blockDim - ������� ����� �������
// gridDim - ������� ����� ������ �������
__global__ void negative(uchar* img, int channel, int N)
{
    int i = 3 * (blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= N)
        return;
    img[i + channel] = 255 - img[i + channel];
}

int LaunchGPU(int channel, string path)
{
    // ����� ���������� �� ���������
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);  // ���������� ���������� � ������ ����������
    print_cuda_device_info(prop);       // ����� ���������� �� ����������� ����������

    // ������ �����
    Mat image = imread(path);
    if (!image.data)    // �������� ����������� �����
    {
        cout << "Could not open or find the image" << std::endl;
        return -1;
    }

    // ���������� ������ �������
    cudaEvent_t start, stop;
    float gpuTime = 0.0f;

    // ������� ������ � ��������� ���������� ����
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // ��������� ����������� ������ ��������� �����������
    int _width = image.cols;
    int _height = image.rows;
    int _type = image.type();

    // ������ � ������������
    uchar* host_img = image.data;                                                       // �������� ��������� �� ������
    size_t N = image.rows * image.cols * 3;                                             // ���������� N - ���������� ���� rgb ������� (� ������� ������� �� �� 3)

    uchar* dev_img;                                                                     // ���������� ���������� ��� �����������

    CHECK(cudaMalloc((void**)&dev_img, N * sizeof(uchar)));                             // ��������� ������ �� ����������

    CHECK(cudaMemcpy(dev_img, host_img, N * sizeof(uchar), cudaMemcpyHostToDevice));    // �������� �������� �� ����������

    int countBlocks = (N / 3 + 511) / 512;                                                  // ���������� ������������ ������
    int countThreads = 512;                                                             // ���������� ����� ��� ������� �����
    printf("Blocks: %i\t Threads: %i\n", countBlocks, countThreads);

    cudaEventRecord(start, 0);                                                          // ����������� start � �������� �����

    negative KERNEL_ARGS2(countBlocks, countThreads) (dev_img, channel, N);             // ������ negative() �� ���� GPU

    cudaEventRecord(stop, 0);                                                           // ����������� stop � �������� �����
    cudaEventSynchronize(stop);                                                         // ���������� ��������� ��������� ���������� ����, ��������� ����������� ������������� �� ������� stop
    cudaEventElapsedTime(&gpuTime, start, stop);                                        // ����������� ����� ����� ��������� start � stop
    printf("Time on GPU = %f milliseconds\n", gpuTime);                                 // ����� ����� ���������� � �������������
    cudaEventDestroy(start);                                                            // ������� ������� start
    cudaEventDestroy(stop);                                                             // ������� ������� stop

    CHECK(cudaGetLastError());                                                          // �������� �� ������

    CHECK(cudaMemcpy(host_img, dev_img, N * sizeof(uchar), cudaMemcpyDeviceToHost));    // �������� �������� � ����������

    Mat imageOut = Mat(_height, _width, _type, host_img);                               // ����������� ���������� uchar � Mat

    CHECK(cudaFree(dev_img));                                                           // �������

    // ���������� �����������
    imwrite("out_img_gpu.jpg", imageOut);

    //// ����� �����������
    //namedWindow("Display window", WINDOW_AUTOSIZE); // �������� ���� �����������
    //imshow("Display window", imageOut);             // ������ ���� ���������� �����������
    //waitKey(0);                                     // �������� ������� ������� � ����
    
    return 0;
}