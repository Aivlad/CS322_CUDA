
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "cu.cuh"

using namespace cv;
using namespace std;

// GPU: применение к заданному каналу изображени€ операции "негатив"
// threadIdx - координаты потока в блоке потоков
// blockIdx - координаты блока потоков в сетке
// blockDim - размеры блока потоков
// gridDim - размеры сетки блоков потоков
__global__ void negative(uchar* img, int channel, int N)
{
    int i = 3 * (blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= N)
        return;
    img[i + channel] = 255 - img[i + channel];
}

int LaunchGPU(int channel, string path)
{
    // ¬ывод информации об устрйосве
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);  // извлечение информации о нужном устройстве
    print_cuda_device_info(prop);       // вывод информации об извлеченном устрйостве

    // „тение файла
    Mat image = imread(path);
    if (!image.data)    // проверка корректного входа
    {
        cout << "Could not open or find the image" << std::endl;
        return -1;
    }

    // ѕеременные замера времени
    cudaEvent_t start, stop;
    float gpuTime = 0.0f;

    // —обыти€ начала и окончани€ выполнени€ €дра
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // —охран€ем необходимые данные исходного изображени€
    int _width = image.cols;
    int _height = image.rows;
    int _type = image.type();

    // –абота с изображением
    uchar* host_img = image.data;                                                       // получаем указатель на данные
    size_t N = image.rows * image.cols * 3;                                             // определ€ем N - количество всех rgb каналов (у каждого пиксел€ их по 3)

    uchar* dev_img;                                                                     // переменна€ устройства дл€ изображени€

    CHECK(cudaMalloc((void**)&dev_img, N * sizeof(uchar)));                             // выделение пам€ти на устройстве

    CHECK(cudaMemcpy(dev_img, host_img, N * sizeof(uchar), cudaMemcpyHostToDevice));    // копируем значение на устройство

    int countBlocks = (N / 3 + 511) / 512;                                                  // количество параллельных блоков
    int countThreads = 512;                                                             // количество нитей дл€ каждого блока
    printf("Blocks: %i\t Threads: %i\n", countBlocks, countThreads);

    cudaEventRecord(start, 0);                                                          // прив€зываем start к текущему месту

    negative KERNEL_ARGS2(countBlocks, countThreads) (dev_img, channel, N);             // запуск negative() на €дре GPU

    cudaEventRecord(stop, 0);                                                           // прив€зываем stop к текущему месту
    cudaEventSynchronize(stop);                                                         // дожидаемс€ реального окончани€ выполнени€ €дра, использу€ возможность стнхронизации по событию stop
    cudaEventElapsedTime(&gpuTime, start, stop);                                        // запрашиваем врем€ между событи€ми start и stop
    printf("Time on GPU = %f milliseconds\n", gpuTime);                                 // здесь сразу вычисление в миллисекундах
    cudaEventDestroy(start);                                                            // убиваем событие start
    cudaEventDestroy(stop);                                                             // убиваем событие stop

    CHECK(cudaGetLastError());                                                          // проверка на ошибки

    CHECK(cudaMemcpy(host_img, dev_img, N * sizeof(uchar), cudaMemcpyDeviceToHost));    // копируем значение с устройства

    Mat imageOut = Mat(_height, _width, _type, host_img);                               // преобразуем полученный uchar в Mat

    CHECK(cudaFree(dev_img));                                                           // очистка

    // —охранение изображени€
    imwrite("out_img_gpu.jpg", imageOut);

    //// ѕоказ изображени€
    //namedWindow("Display window", WINDOW_AUTOSIZE); // —оздани€ окна отображени€
    //imshow("Display window", imageOut);             // ¬нутри окна показываем изображение
    //waitKey(0);                                     // ќжидание нажати€ клавиши в окне
    
    return 0;
}