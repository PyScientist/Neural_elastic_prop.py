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
        self.print_predicted_csv()

    def print_predicted_csv(self):
        file_path_to_write = f'./predicted_{self.parameter}_.csv'
        text = ''
        for num_y, val_y in enumerate(self.y_pred):
            if num_y != len(self.y_pred)+1:
                text += f'{num_y},{val_y}\n'
            else:
                text += f'{num_y},{val_y}'

        with open(file_path_to_write, 'w', encoding='utf-8') as f:
            f.write(text)

        print(f'Predicted list into file: {file_path_to_write}')



