from preprocessing import InitData
from preprocessing import TrainedNets
from processing import ProcessedData
from console_messages import InitialPrintDialog
from console_messages import LoadingAndPreparingNetDialog


def main():

    while True:
        InitialPrintDialog()
        LoadingAndPreparingNetDialog.loading()
        init_set = InitData()  # Готовим данные для обучения
        LoadingAndPreparingNetDialog.loaded(init_set.data_set_to_learn_df.shape)

        LoadingAndPreparingNetDialog.processing()
        trained_nets = TrainedNets(init_set.data_set_to_learn_df)  # Отдаем таблицу на обучение и получаем модели
        metrics1 = (trained_nets.model1['r2_score'], trained_nets.model1['RMSE'], trained_nets.model1['MAE'])
        metrics2 = (trained_nets.model2['r2_score'], trained_nets.model2['RMSE'], trained_nets.model2['MAE'])
        LoadingAndPreparingNetDialog.processed(metrics1, metrics2)

        LoadingAndPreparingNetDialog.chose_processing_data()
        data_path = input('Введите путь к файлу с данными для обработки:')

        predicting1_init = ProcessedData(data_path, trained_nets.model1, 'Модуль упругости при растяжении')
        predicting1_init.predict()

        predicting2_init = ProcessedData(data_path, trained_nets.model2, 'Прочность при растяжении')
        predicting2_init.predict()
        break

if __name__ == '__main__':
    main()
