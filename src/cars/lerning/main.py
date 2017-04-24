# -*- coding: utf-8 -*-
# <autor>Кныш Александр</autor>

from Training import *
from module.Dynamic.Dynamic import Dynamic
from Test import *
from genericModel import *


def main(arg):
    '''
    Старотовый метод приложения.
    :param arg: Объект с параметрами запуска проложения.
    :return: 0 в случае успеха иначе код ошибки
    '''
    print '---start---'

    # Конфигурационный объект
    conf = Dynamic()
    # размеры изображений.
    conf.img_width, conf.img_height = 100, 100
    # пути к обучающим и валидационным сетам
    conf.train_data_dir = '/home/aknysh/git/DiplomVision/src/cars/lerning/data/train'
    conf.validation_data_dir = '/home/aknysh/git/DiplomVision/src/cars/lerning/data/validation'
    #  выходные сети (по эпохам)
    conf.weights_path = '/home/aknysh/git/DiplomVision/src/cars/lerning/res/forwardBack'
    # директория с тестовыми данными.
    conf.path_test = '/home/aknysh/git/DiplomVision/src/cars/lerning/data/validation'
    conf.nb_train_samples = 10600  # количестко самплев обучения
    conf.nb_validation_samples = 4680  # количестко самплев валидации
    conf.epochs = 100  # кол-во всего эпох
    conf.batch_size = 10
    # c какой эпохи начинать обучение...
    conf.ep = 1
    # обученная сеть
    conf.ns = '/home/aknysh/git/DiplomVision/src/cars/lerning/res_100x100_60/forwardBack_100.h5'

    # инициализация модели
    model = create_model(img_height=conf.img_width, img_width=conf.img_height)

    if (arg.isTrain):
        train(model=model, conf=conf)
    else:
        test(model=model, conf=conf)
    print('---end---\n')
    pass


# отправная точка приложения.
if __name__ == '__main__':
    arg = Dynamic()
    arg.isTrain = False
    main(arg)
