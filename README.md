## Учебные проекты программы обучения "Курс по Data Science"

### [temperature_prediction](https://github.com/denis-42ds/innostage_labs/tree/innostage/temperature_prediction)
- Необходимо предсказать температуру на каждый день в течение следующих 730 дней.
- Для выполнения требуется создать файл, в котором будет одна колонка Temperature, в которой содержатся 730 предсказаний.
#### Статус проекта: завершён
#### Использованные инструменты: 
- python, pandas, numpy, matplotlib, seaborn, statsmodels, prophet, sarima
#### Заключение:
- обучены две модели: FBProphet, SARIMA
- более высокие показатели метрик оказались у SARIMA

### [creating_the_serverloadanalyzer_class](https://github.com/denis-42ds/innostage_labs/tree/innostage/creating_the_serverloadanalyzer_class)
- Необходимо создать класс ServerLoadAnalyzer для анализа распределения нагрузки на сервере с учетом времени и оценки параметров распределения Пуассона.
- Необходимо проверить работоспособность созданного класса
#### Статус проекта: завершён
#### Использованные инструменты:
- python, datetime, numpy, scipy
#### Заключение:
- создан класс ServerLoadAnalyzer с указанными параметрами
- проведено тестирование созданного класса ServerLoadAnalyzer
- полученное значение λ соответствует действительности
