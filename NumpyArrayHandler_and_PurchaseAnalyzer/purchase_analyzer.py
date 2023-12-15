import pandas as pd
import matplotlib.pyplot as plt

class PurchaseAnalyzer:
    '''
    анализирует данные о покупках курсов в csv файле с полями name, price и date
    '''
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
    
    def popular_courses_report(self, top):
        '''
        загружает данные из указанного csv файла
        '''
        popular_courses = self.data['name'].value_counts().head(top)
        return popular_courses
    
    def plot_popular_courses(self, top):
        '''
        определяет, какие курсы покупаются чаще всего, 
        и выводит топ N (где N равно top) самых популярных курсов.
        '''
        popular_courses = self.popular_courses_report(top)
        popular_courses.plot(kind='bar')
        plt.xlabel('Course')
        plt.ylabel('Number of Purchases')
        plt.title('Popular Courses')
        plt.show()