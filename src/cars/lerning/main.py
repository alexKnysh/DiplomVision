# -*- coding: utf-8 -*-
# <autor>Кныш Александр</autor>


from Training import *
from module.Dynamic.Dynamic import Dynamic
from slidingWindow.slidingWindow import *
from Test import *
from genericModel import *

from src.cars.lerning.out_excel import create_log


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
    conf.img_width, conf.img_height = 24, 24
    # пути к обучающим и валидационным сетам
    conf.train_data_dir = '/home/aknysh/db/data/train'
    conf.validation_data_dir = '/home/aknysh/db/data/validation'
    #  выходные сети (по эпохам)
    conf.weights_path = '/home/aknysh/git/DiplomVision/src/cars/lerning/cars/30x30/cars'
    # директория с тестовыми данными.
    conf.path_test = '/home/aknysh/db/data/validation/'
    conf.testCount = 1231
    conf.nb_train_samples = 100  # количестко самплев обучения
    conf.nb_validation_samples = 1231  # количестко самплев валидации
    conf.epochs = 100  # кол-во всего эпох
    conf.batch_size = 10
    # c какой эпохи начинать обучение...
    conf.ep = 1
    # обученная сеть
    conf.ns = '/home/aknysh/git/DiplomVision/src/cars/lerning/cars/30x30/cars_3.h5'

    # инициализация модели
    model = create_model(img_height=conf.img_width,
                         img_width=conf.img_height,
                         path_save=conf.weights_path)
    if (arg.isTrain):
        train(model=model, conf=conf)
        create_log(conf=conf)
    else:
        model.load_weights(conf.ns)
        model.compile(loss='binary_crossentropy',
                      optimizer='rmsprop',
                      metrics=['accuracy',
                               precision_threshold(0.5),
                               recall_threshold(0.5)])
        print('Model is compiled\n')
        test(model=model, conf=conf)

        print('---end---\n')
        pass  # отправная точка приложения.


if __name__ == '__main__':
    arg = Dynamic()
    arg.isTrain = True
    main(arg)
