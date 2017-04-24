# -*- coding: utf-8 -*-
# <autor>Кныш Александр</autor>

from Training import *
from module.Dynamic.Dynamic import Dynamic
from slidingWindow.slidingWindow import *
from Test import *
from genericModel import *
import cv2


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
    conf.train_data_dir = '/home/aknysh/db/data/train'
    conf.validation_data_dir = '/home/aknysh/db/data/validation'
    #  выходные сети (по эпохам)
    conf.weights_path = '/home/aknysh/git/DiplomVision/src/cars/lerning/res/forwardBack'
    # директория с тестовыми данными.
    conf.path_test = '/home/aknysh/git/DiplomVision/src/cars/lerning/data/validation'
    conf.nb_train_samples = 20372  # количестко самплев обучения
    conf.nb_validation_samples = 12311  # количестко самплев валидации
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
        # test(model=model, conf=conf)
        t = SlidingWindow(model=model,
                          width=36,
                          height=36,
                          step_x=36,
                          step_y=36,
                          delta=25)
        img = cv2.imread('/home/aknysh/git/DiplomVision/src/cars/lerning/data/test/001099_5.jpg')
        cv2.imshow('orig', img)
        t.sliding_windows(test_img=img,
                          img_width=conf.img_width,
                          img_height=conf.img_height,
                          path_dir='/home/aknysh/git/DiplomVision/src/cars/lerning/data/add_bad',
                          ns=conf.ns)
    print('---end---\n')
    pass


# отправная точка приложения.
if __name__ == '__main__':
    arg = Dynamic()
    arg.isTrain = True
    main(arg)
