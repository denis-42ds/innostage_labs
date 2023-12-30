import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

class CreditDataAnalysis:
	'''
 Осуществляет анализ кредитных данных с использованием методов машинного обучения. 
 Включает функции загрузки набора данных, предварительной обработки категориальных признаков, 
 разделения данных, масштабирования признаков, обучения модели и оценки ее производительности.
 '''
	def __init__(self):
		self.data = None
		self.X_train = None
		self.X_test = None
		self.y_train = None
		self.y_test = None
		self.model = LogisticRegression(max_iter=1000)
		self.accuracy_before_scaling = None
		self.accuracy_after_scaling = None
		self.scaling_effect_percentage = None
		
	def load_credit_dataset(self):
		'''
  загрузка датасета с данными (использована версия 2)
  '''
		credit_data = fetch_openml(name='credit-g', parser="auto", version=2)
		self.data = pd.DataFrame(data=credit_data.data, 
								 columns=credit_data.feature_names)
		self.data['target'] = credit_data.target
		
	def encode_data(self, encoder=None):
		'''
  - преобразовывает все категориальные признаки в числа, используя энкодер 
  - удаляет старые признаки из self.data, 
  - добавляет преобразованные
  - энкодер может быть как LabelEncoder, так и OneHotEncoder
  '''
		categorical_features = self.data.select_dtypes(include='category').columns

		if encoder is None:
			encoder = LabelEncoder()
			
		if isinstance(encoder, LabelEncoder):
			for feature in categorical_features:
				self.data[feature] = encoder.fit_transform(self.data[feature])

		elif isinstance(encoder, OneHotEncoder):
			encoded_features = encoder.fit_transform(self.data[categorical_features])
			feature_names = encoder.get_feature_names_out(categorical_features)
			encoded_df = pd.DataFrame(encoded_features.toarray(), columns=feature_names)
			self.data = pd.concat([self.data.drop(categorical_features, axis=1), encoded_df], axis=1)
		
	def split_data(self, test_size=0.2):
		'''
  - выделяет из датафрейма целевой признак
  - разделяет датафрейм на тренировочную и тестовую выборки
  '''
		X = self.data.drop('target', axis=1)
		y = self.data['target']
		self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, 
																				y, 
																				test_size=test_size, 
																				random_state=42)
		
	def scale_features(self, scaler=StandardScaler()):
		'''
  производит масштабирование признаков
  '''
		self.X_train = pd.DataFrame(scaler.fit_transform(self.X_train), 
									columns=self.X_train.columns, 
									index=self.X_train.index)
		self.X_test = pd.DataFrame(scaler.transform(self.X_test), 
								   columns=self.X_test.columns, 
								   index=self.X_test.index)
		
	def train_model(self, model=LogisticRegression(max_iter=1000)):
		'''
  производит обучение модели на тренировочной выборке
  '''
		self.model = model
		self.model.fit(self.X_train, self.y_train)
		
	def evaluate_model(self, metric):
		'''
  производит оценку качества модели
  '''
		y_pred = self.model.predict(self.X_test)
		return metric(self.y_test, y_pred)
	
	def compare_scaling_effect(self, 
							   scaler=StandardScaler(),
							   model=LogisticRegression(max_iter=1000), 
							   encoder=LabelEncoder(), 
							   metric=accuracy_score):
		'''
  выявляет эффект от масштабирования признаков
  '''
		self.load_credit_dataset()
		self.encode_data(encoder)
		self.split_data()
		# Обучение модели до масштабирования
		self.train_model(model)
		self.accuracy_before_scaling = self.evaluate_model(metric)
		# Масштабирование и обучение модели после
		self.scale_features(scaler)
		self.train_model()
		self.accuracy_after_scaling = self.evaluate_model(metric)
        # Вычисление процента изменения
		self.scaling_effect_percentage = ((self.accuracy_after_scaling - 
										   self.accuracy_before_scaling) / 
										  self.accuracy_before_scaling) * 100
        # Вывод результатов
		print(f'Accuracy before scaling: {self.accuracy_before_scaling}')
		print(f'Accuracy after scaling: {self.accuracy_after_scaling}')
		print(f'Scaling effect percentage: {self.scaling_effect_percentage:.2f}%')