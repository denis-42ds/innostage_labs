# Название проекта: Анализ мошенничества с кредитными картами

## Статус проекта: в работе

## Описание проекта:

1. Корреляционный анализ: 
  - Рассчитать корреляционную матрицу для числовых переменных. 
  - Возвращаемый тип данных - numpy массив с размерностью 2.
2. Проверка статистических гипотез: 
  - Проверить гипотезы, например, о равенстве средних сумм для признаков. 
  - Параметры включают признаки для сравнения, критерий проверки (t-test, z-test, kolmogorov-smirnov, fisher) и уровень значимости.
3. Точечные оценки параметров распределения: 
  - Оценить средние и дисперсии для выбранных признаков с использованием метода моментов и метода максимального правдоподобия. 
  - Возвращаемый тип данных - строка с оценками.
4. Определение влияния признаков на безопасность транзакции: 
  - Применить логистическую регрессию для выявления признаков, оказывающих наибольшее влияние на безопасность транзакций. 
  - Возвращаемый тип данных - список с названиями признаков на основе весов в логистической регрессии, ограниченный top_k параметрами.
5. Интервальные оценки: 
  - Рассчитать доверительные интервалы для среднего значения выбранного признака с учетом уровня значимости. 
  - Использовать t- или z-статистику в зависимости от размера выборки (подсказка: центральная предельная теорема и значение 30). 
  - Возвращаемый тип данных - tuple с интервальной оценкой в формате (lower_bound, upper_bound).

## Цель проекта:

#### Использованные инструменты:

- python, pandas, numpy

#### Заключение:
- 