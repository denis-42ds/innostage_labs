# Название проекта: Создание и проверка аналитических классов

## Статус проекта: Завершён

## Описание проекта:

1. Класс для работы с массивом Numpy:
  <br>Необходимо написать класс NumpyArrayHandler,
  <br>который принимает на вход массив Numpy любой размерности.
  <br>Класс должен содержать следующие методы:
    - `__init__(self, array: np.ndarray)`: Конструктор класса, <br>который принимает массив Numpy и сохраняет его внутри объекта.
    - `get_dimensions(self) - int`: Метод, который возвращает количество размерностей (осей) массива.
    - `calculate_sum(self) - float`: Метод, который возвращает сумму всех элементов массива.

2. Класс PurchaseAnalyzer, который предназначен для анализа данных <br>о покупках курсов в CSV-файле с полями name, price и date.
  <br>Класс должен выполнять следующие задачи:
    - Загрузка данных: Класс должен иметь конструктор, который загружает данные из указанного CSV-файла.
    - Отчёт о популярных курсах: метод `popular_courses_report(top)`,
    <br>который определяет, какие курсы покупаются чаще всего, <br>и выводит топ N (где N равно top) самых популярных курсов.
  <br>В этом классе можно использовать Pandas и Matplotlib, если необходимо.

## Цель проекта: Улучшение навыков по созданию классов

## Использованные инструменты:

- python, pandas, numpy, matplotlib

## Заключение:
- созданы два класса в отдельных py файлах
- проверка показала, что с поставленными задачами написанные классы успешно справляются
