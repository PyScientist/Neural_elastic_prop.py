import pandas as pd


class ProcessedData:
    def __init__(self, path_data, model, parametr):
        self.parameter = parametr
        self.path_data = path_data
        self.model = model
        self.X = pd.read_csv(path_data, sep=';', engine='python').values
        self.X = self.model['scaler'].transform(self.X)
        self.y_pred = None


    def predict(self):
        self.y_pred = self.model['model'].predict(self.X)
        print(f'Получен следующий массив результирующих данных по модели для параметра: {self.parameter}')
        print(list(self.y_pred))


