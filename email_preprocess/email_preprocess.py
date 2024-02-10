import pandas as pd
from sklearn.model_selection import train_test_split

class EmailPreprocess:
    def __init__(self, filename):
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
        self.feature_selection_methods = []
    def load_data(self):
        self.data = pd.read_csv(self.filename)
    def check_missing_values(self):
        # Проверить на отсутствие значений и обработать их, если они есть
        pass
    def split_dataset(self):
        self.X = self.data.iloc[:, 1:-1]
        self.y = self.data.iloc[:, -1]
        # Извлечь признаки
        # Извлечь меткиdef feature_scaling(self):
        pass
    def train_test_split(self, test_size=0.2, random_state=42):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
        self.X, self.y, test_size=test_size, random_state=random_state)
    def feature_selection(self, method=, k=):
        pass
    def text_cleaning(self):
        pass
    def preprocess(self):
        self.load_data()
        self.check_missing_values()
        self.split_dataset()
        self.text_cleaning()
        self.feature_scaling()
        self.train_test_split()