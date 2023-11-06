from datetime import datetime, timedelta

class ServerLoadAnalyzer:
	def __init__(self):
		self.requests = []
		
	def add_request(self, timestamp, load):
		'''
  Добавляет данные о запросе на сервер. 
  timestamp - метка времени (в формате datetime), 
  load - нагрузка в данном запросе
		'''
		self.requests.append((timestamp, load))

	
	def estimate_poisson_parameters(self, time_delta=3600):
		'''
  Анализирует данные и оценивает параметры распределения Пуассона 
  за заданный интервал времени (time_delta в секундах, по умолчанию 3600 секунд). 
  Возвращает словарь с оцененными параметрами: {"lambda": lambda_value}.
		'''
		end_time = max(timestamp for timestamp, _ in self.requests)
		start_time = end_time - timedelta(seconds=time_delta)
		filtered_requests = [(timestamp, load) for timestamp, load in self.requests if start_time <= timestamp <= end_time]
				
		if len(filtered_requests) == 0:
			return {"lambda": 0}
			
		total_load = sum(load for _, load in filtered_requests)
		lambda_value = total_load / (time_delta / 3600) / len(filtered_requests) # приведение к часам, деление на количество запросов

		return {"lambda": lambda_value}