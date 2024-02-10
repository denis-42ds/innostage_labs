## Учебные проекты программы обучения "Курс по Data Science"

### [Ссылка на теоретические материалы](https://cloud.mail.ru/public/2DWe/mwFSv4coG)

### [email_preprocess](https://github.com/denis-42ds/innostage_labs/tree/innostage/email_preprocess)
#### Задачи:
- создание класса для предобработки данных, предназначенных для классификации электронных писем на "спам" и "не спам"
#### Статус проекта: завершён
#### Использованные инструменты:
- python, pandas, sklearn
#### Заключение:
- создан класс в отдельном .py файле
- проверка показала, что с поставленными задачами написанный класс успешно справляется

### [credit_data_analysis](https://github.com/denis-42ds/innostage_labs/tree/innostage/credit_data_analysis)
#### Задачи:
- создание класса для анализа кредитных данных с использованием методов машинного обучения
#### Статус проекта: завершён
#### Использованные инструменты:
- python, pandas, sklearn
#### Заключение:
- создан класс в отдельном .py файле
- проверка показала, что с поставленными задачами написанный класс успешно справляется

### [NumpyArrayHandler_and_PurchaseAnalyzer](https://github.com/denis-42ds/innostage_labs/tree/innostage/NumpyArrayHandler_and_PurchaseAnalyzer)
#### Задачи:
- создание класса для анализа массивов
- создание класса для первичного анализа данных о покупках
#### Статус проекта: завершён
#### Использованные инструменты:
- python, pandas, numpy, matplotlib
#### Заключение:
- созданы два класса в отдельных py файлах
- проверка показала, что с поставленными задачами написанные классы успешно справляются

### [credit_card_fraud_analysis](https://github.com/denis-42ds/innostage_labs/tree/innostage/credit_card_fraud_analysis)
#### Задачи:
- Корреляционный анализ
- Проверка статистических гипотез
- Точечные оценки параметров распределения
- Определение влияния признаков на безопасность транзакции
- Интервальные оценки
#### Статус проекта: завершён
#### Использованные инструменты:
- python, pandas, numpy, scipy, sklearn, statsmodels, seaborn, matplotlib, phik
#### Заключение:
- создан класс CreditCardFraudAnalysis, методы которого производят:
  + Корреляционный анализ
  + Проверка статистических гипотез
  + Точечные оценки параметров распределения
  + Определение влияния признаков на безопасность транзакции
  + Интервальные оценки
- произведена проверка работоспособности класса

### [server_load_analyzer](https://github.com/denis-42ds/innostage_labs/tree/innostage/server_load_analyzer)
- Необходимо создать класс ServerLoadAnalyzer для анализа распределения нагрузки на сервере с учетом времени и оценки параметров распределения Пуассона.
- Необходимо проверить работоспособность созданного класса
#### Статус проекта: завершён
#### Использованные инструменты:
- python, datetime
#### Заключение:
- создан класс ServerLoadAnalyzer с указанными параметрами
- проведено тестирование созданного класса ServerLoadAnalyzer
- полученное значение λ соответствует действительности

### [temperature_prediction](https://github.com/denis-42ds/innostage_labs/tree/innostage/temperature_prediction)
- Необходимо предсказать температуру на каждый день в течение следующих 730 дней.
- Для выполнения требуется создать файл, в котором будет одна колонка Temperature, в которой содержатся 730 предсказаний.
#### Статус проекта: завершён
#### Использованные инструменты: 
- python, pandas, numpy, matplotlib, seaborn, statsmodels, prophet, sarima
#### Заключение:
- обучены две модели: FBProphet, SARIMA
- более высокие показатели метрик оказались у SARIMA
