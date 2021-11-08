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
    // Чтение файла
    Mat image = imread(path);
    if (!image.data)    // проверка корректного входа
    {
        cout << "Could not open or find the image" << std::endl;
        return -1;
    }

    // Переменные замера времени
    clock_t start;
    double cpuTime = 0.0;

    start = clock();
    // Работа и изображением
    for (int i = 0; i < image.rows; i++)
    {
        Vec3b* p = image.ptr<Vec3b>(i);             // указатель на первый пиксель в строке

        for (int j = 0; j < image.cols; j++)
        {
            p[j][channel] = 255 - p[j][channel];    // негатив на указанный канал
        }
    }
    cpuTime = (double)(clock() - start) / CLOCKS_PER_SEC;
    printf("Time on CPU = %f milliseconds (original: %f)\n", (cpuTime * 1000), cpuTime);    // *1000 т.к. секунды переводим в миллисекунды

    // Сохранение изображения
    imwrite("out_img_cpu.jpg", image);

    //// Показ изображения
    //namedWindow("Display window", WINDOW_AUTOSIZE); // Создания окна отображения
    //imshow("Display window", imageOut);             // Внутри окна показываем изображение
    //waitKey(0);                                     // Ожидание нажатия клавиши в окне

	return 0;
}