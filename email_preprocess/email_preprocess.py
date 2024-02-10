import re
import string
import pandas as pd

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import (
    SelectKBest, 
    f_classif, 
    mutual_info_classif, 
    chi2, 
    f_regression, 
    mutual_info_regression
    )

RANDOM_STATE = 42

class EmailPreprocess:
    '''
    предназначен для обработки и предварительной обработки
    данных, предназначенных для классификации электронных писем на спам и не спам
    '''
    def __init__(self, filename):
        '''
        принимает имя файла с данными в качестве аргумента и инициализирует атрибуты класса
        '''
        self.filename = filename
        self.data = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = None
        self.feature_selector = None
        self.feature_selection_methods = {
            'f_classif': f_classif,
            'mutual_info_classif': mutual_info_classif,
            'chi2': chi2,
            'f_regression': f_regression,
            'mutual_info_regression': mutual_info_regression
        }
		
    def load_data(self):
        '''
        загружает данные из CSV-файла в pandas DataFrame
        '''
        self.data = pd.read_csv(self.filename)
        print('Данные загружены')
        		
    def check_missing_values(self):
        '''
        проверяет наличие отсутствующих значений в данных и обрабатывает их при необходимости
        '''
        missing_values = self.data.isnull().sum()
        features_with_missing_values = missing_values[missing_values > 0].index.tolist()
        if len(features_with_missing_values) > 0:
            print("Признаки с пропущенными значениями:")
            print(features_with_missing_values)
            self.X[features_with_missing_values] = (
				self.X[features_with_missing_values]
				.fillna(self.X[features_with_missing_values].median())
			)
            print('Пропуски заполнены медианными значениями')
        else:
            print('Пропусков не обнаружено')
		
    def split_dataset(self):
        '''
        разделяет набор данных на признаки (X) и метки (y)
        '''
        self.X = self.data.iloc[:, 1:-1]
        self.y = self.data.iloc[:, -1]
        print('Набор данных разделён на признаки и метки')
        
    def text_cleaning(self):
        '''
        очищает текст электронных писем от лишних символов, пунктуации и стоп-слов
        '''
        stop_words = set(stopwords.words('english'))
        self.X['text'] = self.X['text'].apply(lambda x: self.clean_text(x, stop_words) if isinstance(x, str) else x)
        print('Текст писем очищен от лишних символов, пунктуации и стоп-слов')
        
    def clean_text(self, text, stop_words):
        '''
        вспомогательный метод для text_cleaning
        '''
        # Удаление пунктуации
        text = text.translate(str.maketrans('', '', string.punctuation))
        # Приведение к нижнему регистру
        text = text.lower()
        # Удаление стоп-слов
        text = ' '.join([word for word in word_tokenize(text) if word not in stop_words])
        # Удаление чисел
        text = re.sub(r'\d+', '', text)
        # Удаление лишних пробелов
        text = re.sub(r'\s+', ' ', text).strip()
        return text
        
    def feature_scaling(self):
        '''
        масштабирует признаки, чтобы они имели сравнимые диапазоны значений
        '''
        self.scaler = StandardScaler()
        self.X = self.scaler.fit_transform(self.X)
        print('Произведено масштабирование признаков')
		
    def train_test_split(self, test_size=0.2, random_state=RANDOM_STATE):
        '''
        разделяет данные на обучающий и тестовый наборы
        '''
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
        self.X, self.y, test_size=test_size, random_state=random_state)
        print('Данные разделены на обучающий и тестовый наборы')
		
    def feature_selection(self, method='f_classif', k=None):
        '''
        - выбирает наиболее значимые признаки 
        - Пользователь может выбрать метод (из списка self.feature_selection_methods) и количество признаков (k)
        '''
        if method in self.feature_selection_methods:
            self.feature_selector = SelectKBest(self.feature_selection_methods[method], k=k)
            self.X_train = self.feature_selector.fit_transform(self.X_train, self.y_train)
            self.X_test = self.feature_selector.transform(self.X_test)
            print('Отобраны наиболее значимые признаки')
        else:
            raise ValueError(f"Invalid feature selection method: {method}")
		
    def preprocess(self):
        '''
        последовательно вызывает все остальные методы для предварительной обработки данных
        '''
        self.load_data()
        self.check_missing_values()
        self.split_dataset()
        self.text_cleaning()
        self.feature_scaling()
        self.train_test_split()
        print('Произведена предварительная обработка данных')