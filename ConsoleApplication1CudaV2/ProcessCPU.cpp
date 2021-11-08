#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

#include <stdio.h>
#include <string.h>
#include <memory.h>

using namespace cv;
using namespace std;

int LaunchCPU(int channel, string path)
{
    // ������ �����
    Mat image = imread(path);
    if (!image.data)    // �������� ����������� �����
    {
        cout << "Could not open or find the image" << std::endl;
        return -1;
    }

    // ���������� ������ �������
    clock_t start;
    double cpuTime = 0.0;

    start = clock();
    // ������ � ������������
    for (int i = 0; i < image.rows; i++)
    {
        Vec3b* p = image.ptr<Vec3b>(i);             // ��������� �� ������ ������� � ������

        for (int j = 0; j < image.cols; j++)
        {
            p[j][channel] = 255 - p[j][channel];    // ������� �� ��������� �����
        }
    }
    cpuTime = (double)(clock() - start) / CLOCKS_PER_SEC;
    printf("Time on CPU = %f milliseconds (original: %f)\n", (cpuTime * 1000), cpuTime);    // *1000 �.�. ������� ��������� � ������������

    // ���������� �����������
    imwrite("out_img_cpu.jpg", image);

    //// ����� �����������
    //namedWindow("Display window", WINDOW_AUTOSIZE); // �������� ���� �����������
    //imshow("Display window", imageOut);             // ������ ���� ���������� �����������
    //waitKey(0);                                     // �������� ������� ������� � ����

	return 0;
}