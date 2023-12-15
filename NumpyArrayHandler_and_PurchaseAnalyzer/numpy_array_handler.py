import numpy as np

class NumpyArrayHandler:
    def __init__(self, array: np.ndarray):
        self.array = array
    
    def get_dimensions(self) -> int:
        '''
        возвращает количество размерностей (осей) массива
        '''
        return self.array.ndim
    
    def calculate_sum(self) -> float:
        '''
        возвращает сумму всех элементов массива
        '''
        return np.sum(self.array)