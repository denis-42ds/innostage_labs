import pandas as pd
import numpy as np
from scipy.stats import ttest_ind, ks_2samp, fisher_exact, t, norm, sem
from sklearn.linear_model import LogisticRegression
from statsmodels.stats.weightstats import ztest

class CreditCardFraudAnalysis:
	def __init__(self, data: pd.DataFrame):
		'''
		Инициализация класса с данными.
		Параметры:
		data (pd.DataFrame): Набор данных для анализа.
		'''
		self.data = data
		self.features = data.drop(columns=['Class'])
		self.target = data['Class']

	def correlation_task(self) -> np.ndarray:		
		'''
		Рассчитать корреляционную матрицу для числовых переменных. (ndim == 2)
		Возвращает:	
		np.ndarray: Корреляционная матрица.
		'''
		numeric_features = self.features.select_dtypes(include=[np.number])
		correlation_matrix = numeric_features.corr().values

		return correlation_matrix

	def test_hypotheses(self, feature1: str, feature2: str, test_criteria: str, alpha: float) -> str:
		'''		
		Проверка статистических гипотез.
		Параметры:
		feature1 (str): Название первого признака.
		feature2 (str): Название второго признака.
		test_criteria (str): Критерий проверки (t-test, z-test, kolmogorov-smirnov,
		fisher).
		alpha (float): Уровень значимости.
		Возвращает:
		result: Результаты теста в формате '{test_criteria}: верна H0/H1'.
		'''
		data1 = self.data[feature1]
		data2 = self.data[feature2]

        # Проверка выбранного критерия
		if test_criteria == 't-test':
			_, p_value = ttest_ind(data1, data2)
		elif test_criteria == 'z-test':
			_, p_value = ztest(data1, data2)
		elif test_criteria == 'kolmogorov-smirnov':
			_, p_value = ks_2samp(data1, data2)
		elif test_criteria == 'fisher':
			_, p_value = fisher_exact(pd.crosstab(data1, data2))
		else:
			return 'Некорректный критерий'

        # Определение результата теста
		if p_value < alpha:
			result = f'{test_criteria}: верна H1'
		else:
			result = f'{test_criteria}: верна H0'

		return result

	def point_estimates_parameters(self, feature: str) -> str:
		'''
		Точечные оценки параметров распределения.
		Параметры:
		feature (str): Название признака для оценки.
		Возвращает:
		result: Оценки параметров распределения в формате 'Среднее: {mean}, Дисперсия:{var}'.
		'''
		data = self.data[feature]
		mean = data.mean()
		var = data.var()

		result = f'Среднее: {mean}, Дисперсия: {var}'
		return result

	def impact_on_security(self, top_k: int) -> list:
		'''
		Определение влияния признаков на безопасность транзакции.
		Параметры:
		top_k (int): Количество признаков для возвращения с наибольшим влиянием.
		Возвращает:
		result: Признаки с наибольшим влиянием в формате ['V2', 'V6', ...].
		'''
		X = self.features
		y = self.target

		# Обучение логистической регрессии
		model = LogisticRegression()
		model.fit(X, y)

		# Получение коэффициентов влияния признаков
		feature_importances = abs(model.coef_[0])

		# Сортировка признаков по влиянию
		sorted_features = [feature for _, feature in sorted(zip(feature_importances, X.columns), reverse=True)]

		# Возвращение top_k признаков
		result = sorted_features[:top_k]
		return result

	def interval_estimates(self, feature: str, alpha: float) -> tuple:
		'''
		Интервальные оценки.
		Параметры:
		feature (str): Название признака для оценки интервала.
		alpha (float): Уровень значимости.
		Возвращает:
		confidence_interval: Доверительный интервал в формате (val1, val2).
		'''
		data = self.data[feature]
		confidence_interval = norm.interval(1 - alpha, loc=np.mean(data), scale=sem(data))
		lower_bound = confidence_interval[0]
		upper_bound = confidence_interval[1]
		return (lower_bound, upper_bound)
