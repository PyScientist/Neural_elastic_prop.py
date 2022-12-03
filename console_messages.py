from sys import platform
import os

class PlatformTool:
    """Class to adaptation of waiting for any key input for platform"""
    def __init__(self):
        self.platform = 'unknown'
        if platform == "linux" or platform == "linux2":
            self.platform = 'unix'
        elif platform == "darwin":
            self.platform = 'mac'
        elif platform == "win32":
            self.platform = 'windows'

    def next_key_tool(self):
            if self.platform == 'windows':
                import msvcrt
                print("\nНажмите любую кнопку чтобы продолжить...")
                msvcrt.getch()
            elif self.platform == 'unix':
                os.system("/bin/bash -c 'read -s -n 1 -p\"\nНажмите любую кнопку чтобы продолжить...\"'")
            else:
                pass


class InitialPrintDialog:
    """Class for displaying initialization dialog"""
    def __init__(self):
        self.initial_print()

    @staticmethod
    def initial_print():
        """
        Initialization menu
        """
        text_to_print = '\n' \
                        '***********************************************************\n' \
                        '*  Добро пожаловать в диалог анализа свойств композитов:  *\n' \
                        '*                                                         *\n' \
                        '*  Инструмент для прогноза                                *\n' \
                        '*---------------------------------------------------------*\n' \
                        '*  1) Модуля упругости при растяжении, ГПа                *\n' \
                        '*  2) Прочности при растяжении, МПа                       *\n' \
                        '*                                                         *\n' \
                        '*  основываясь на данных 11 параметров                    *\n' \
                        '*                                                         *\n' \
                        '*                                                         *\n' \
                        '***********************************************************\n' \
                        '\n'
        print(text_to_print)
        PlatformTool().next_key_tool()


class LoadingAndPreparingNetDialog:
    """Class for displaying loading and processing dialog"""
    def __init__(self):
        pass

    @staticmethod
    def loading():
        print('\nИдет загрузка..., пожалуйста подождите.')

    @staticmethod
    def loaded(data_shape):
        print(f'\nДанные загружены!!! рамерность данных {data_shape}')
        PlatformTool().next_key_tool()

    @staticmethod
    def processing():
        print('\nИдет подготовка нейронной сети..., пожалуйста подождите.')

    @staticmethod
    def processed(metrics1, metrics2):
        print('\nТренировка моделей выполнена!!! Можно переходить к прогнозированию')
        print(f'Метрики модели для прогнозирования модуля упругости при растяжении: RMSE = {metrics1[1].round(2)}, MAE = {metrics1[2].round(2)}')
        print(f'Метрики модели для прогнозирования прочности при растяжении: RMSE = {metrics2[1].round(2)}, MAE = {metrics2[2].round(2)}')
        PlatformTool().next_key_tool()

    @staticmethod
    def chose_processing_data():
        print('\nТеперь нужно выбрадь данные для обработки, например, "./data_to_process.csv"')
        PlatformTool().next_key_tool()


if __name__ == '__main__':
    pass