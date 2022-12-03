import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor

from sklearn.metrics import mean_squared_error as RMSE
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import r2_score


class InitData:
    def __init__(self):
        self.file_physical_props = './X_bp.xlsx'
        self.file_physical_props_sheet_name = 'X_bp.csv'
        self.file_composite_props = 'X_nup.xlsx'
        self.file_composite_props_sheet_name = 'X_nup.csv'
        self.file_physical_props_df = None
        self.file_composite_props_df = None
        self.data_set_to_learn_df = None

        self.read_data()
        self.join_data()
        self.delete_outliers()

    def read_data(self):
        self.file_physical_props_df = pd.read_excel(io=self.file_physical_props,
                                                    engine='openpyxl',
                                                    sheet_name=self.file_physical_props_sheet_name)
        self.file_composite_props_df = pd.read_excel(io=self.file_composite_props,
                                                     engine='openpyxl',
                                                     sheet_name=self.file_composite_props_sheet_name)

        self.file_physical_props_df.rename(columns={'Unnamed: 0': 'id_phis_props'}, inplace=True)
        self.file_composite_props_df.rename(columns={'Unnamed: 0': 'id_results'}, inplace=True)

    def join_data(self):
        self.data_set_to_learn_df = self.file_composite_props_df.join(self.file_physical_props_df,
                                                                      how='inner',
                                                                      lsuffix='id_results',
                                                                      rsuffix='id_phis_props')

        self.data_set_to_learn_df.drop(columns=['id_phis_props'], inplace=True)
        self.data_set_to_learn_df.drop(columns=['id_results'], inplace=True)

    def delete_outliers(self):
        s = pd.Series(self.data_set_to_learn_df["Угол нашивки, град"])
        categorical_sdf = pd.get_dummies(s)
        self.data_set_to_learn_df = self.data_set_to_learn_df.join(categorical_sdf, how='inner')
        self.data_set_to_learn_df.drop(columns=['Угол нашивки, град'], inplace=True)
        self.data_set_to_learn_df.rename(columns={0: 'Угол нашивки 0'}, inplace=True)
        self.data_set_to_learn_df.rename(columns={90: 'Угол нашивки 90'}, inplace=True)

        # удаляем строку содержащую нулевой шаг нашивки и нулевую плотность нашивки (образец №19)
        self.data_set_to_learn_df = self.data_set_to_learn_df[self.data_set_to_learn_df['Шаг нашивки'] != 0]

        def quantiles_filter_drop(df, column_name):
            'drops_rows_with_outliers'
            min_quan = df[column_name].quantile(0.001)
            max_quan = df[column_name].quantile(0.999)

            df = df[df[column_name] > min_quan]
            df = df[df[column_name] < max_quan]

            return df

        columns = self.data_set_to_learn_df.columns
        for column in columns:
            if ((column != 'Угол нашивки 0') and (column != 'Угол нашивки 90')):
                self.data_set_to_learn_df = quantiles_filter_drop(self.data_set_to_learn_df, column)



class TrainedNets:
    """class for net training based on input data"""

    def __init__(self, dataframe):

        self.dataframe = dataframe

        self.key1 = ['Модуль упругости при растяжении, ГПа']
        self.key2 = ['Прочность при растяжении, МПа']

        self.features1 = ['Шаг нашивки',
                     'Плотность нашивки',
                     'Соотношение матрица-наполнитель',
                     'Плотность, кг/м3',
                     'модуль упругости, ГПа',
                     'Количество отвердителя, м.%',
                     'Содержание эпоксидных групп,%_2',
                     'Температура вспышки, С_2',
                     'Поверхностная плотность, г/м2',
                     'Потребление смолы, г/м2',
                     'Угол нашивки 0',
                     'Угол нашивки 90']

        self.features2 = ['Шаг нашивки',
                     'Плотность нашивки',
                     'Соотношение матрица-наполнитель',
                     'Плотность, кг/м3',
                     'модуль упругости, ГПа',
                     'Количество отвердителя, м.%',
                     'Содержание эпоксидных групп,%_2',
                     'Температура вспышки, С_2',
                     'Поверхностная плотность, г/м2',
                     'Потребление смолы, г/м2',
                     'Угол нашивки 0',
                     'Угол нашивки 90']

        self.X1 = None
        self.X2 = None
        self.y1 = None
        self.y2 = None

        self.prepare_data_for_train()

        self.model1 = self.train_and_validate_model(self.X1, self.y1, 'KNeighborsRegressor', (300))
        self.model2 = self.train_and_validate_model(self.X2, self.y2, 'KNeighborsRegressor', (64))

    def prepare_data_for_train(self):
        self.X1 = self.dataframe[self.features1].values
        self.X2 = self.dataframe[self.features2].values
        self.y1 = self.dataframe[self.key1].values
        self.y2 = self.dataframe[self.key2].values

    @staticmethod
    def train_and_validate_model(X, y, model_type, neuron_model=(300), n_estimators=5):
        # Разбиваем на обучающую и тестовую выборку
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=101, test_size =.3)

        # Делаем плоскими входные данные так изначально они имеют размерность (y,1) преобразуем ее к (y,)
        y_test = y_test.ravel()
        y_train = y_train.ravel()

        if model_type != 'MLPRegressor':
            neuron_model = ''
        pass

        if model_type == 'MLPRegressor':
            model = MLPRegressor(random_state=1, max_iter=7000, hidden_layer_sizes=neuron_model, activation="relu")
        elif model_type == 'RandomForestRegressor':
            model = RandomForestRegressor(max_depth=3, max_features='sqrt', n_estimators=n_estimators, random_state=18)
        elif model_type == 'KNeighborsRegressor':
            model = KNeighborsRegressor(n_neighbors=3)

        # Масштабируем входные данные
        sc = MinMaxScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)

        # Инициализируем модель и обучаем ее
        model.fit(X_train, y_train)

        # Делаем прогноз на тестовых данных
        y_pred = model.predict(X_test)

        # Считаем метрику модели
        r2_score_model = r2_score(y_test, y_pred)
        RMSE_model = RMSE(y_test, y_pred)
        MAE_model = MAE(y_test, y_pred)

        return {'model': model,
                'model_type': model_type,
                'neuron_struct': neuron_model,
                'scaler': sc,
                'r2_score': r2_score_model,
                'RMSE': RMSE_model,
                'MAE': MAE_model,
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test,
                'y_pred': y_pred}



