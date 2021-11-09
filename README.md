# CS322. Программирование ускорителей параллельных вычислителей
*наконец решил все фиксировать, а то и так с кодом долго страдаю (точнее с тем, что он на w10), терять ничего не хотелось бы*  
* *Запись 0*: на данном этапе задача "Применить к заданному каналу изображения операцию негатив" - уже давно выполнена и сдана.
* *Запись 1*: решается задача "Для заданной высокой и узкой (50000x1024) матрицы целых чисел int реализуйте проверку того, что ее строки симметричны относительно средней вертикальной линии. Матрица хранится в памяти по строкам. Функция должна возвращать вектор True/False, содержащий для каждой строки результат проверки на симметрию".  
Состояние: есть действующая проверка на симметрию строк матрицы, результат записывается в вектор int, где 0 – строка матрицы не симметрична и 1 в противоположном случае + сделал подобный вариант через shared memory. 
