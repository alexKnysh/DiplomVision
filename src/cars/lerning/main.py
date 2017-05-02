# -*- coding: utf-8 -*-
# <autor>Кныш Александр</autor>
import os

from xlwt import Workbook

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
    print ('---start---')

    # Конфигурационный объект
    conf = Dynamic()
    # размеры изображений.
    conf.img_width, conf.img_height = 24, 24
    # пути к обучающим и валидационным сетам
    conf.train_data_dir = '/home/aknysh/db/data/train'
    conf.validation_data_dir = '/home/aknysh/db/data/validation'
    #  выходные сети (по эпохам)
    conf.weights_path = '/home/aknysh/git/DiplomVision/src/cars/lerning/cars/24x24_12311'
    # директория с тестовыми данными.
    conf.path_test = '/home/aknysh/db/data/test'
    conf.testCount = 2700
    conf.nb_train_samples = 17669  # количестко самплев обучения
    conf.nb_validation_samples = 12311  # количестко самплев валидации
    conf.epochs = 20  # кол-во всего эпох
    conf.batch_size = 10
    # c какой эпохи начинать обучение...
    conf.ep = 1
    # обученная сеть
    conf.ns = '/home/aknysh/git/DiplomVision/src/cars/lerning/cars/24x24_12311'

    # инициализация модели
    model = create_model(img_height=conf.img_width,
                         img_width=conf.img_height,
                         path_save=conf.weights_path)
    if (arg.isTrain):
        train(model=model, conf=conf)
        create_log(conf=conf)
    else:
        list = os.listdir(conf.ns)
        list = filter(lambda x: x.endswith('h5'), list)
        w = Workbook()
        ws = w.add_sheet('Hey, Dude')
        ws.write(0, 0, u"Название файла")
        ws.write(0, 1, u"Loss - обучение")
        ws.write(0, 2, u"Accuracy - обучение")
        ws.write(0, 3, u"Loss validation - обучение")
        ws.write(0, 4, u"Accuracy validation - обучение")
        ws.write(0, 5, u"Recall")
        ws.write(0, 6, u"Precision")
        ws.write(0, 7, u"F-мера")
        ws.write(0, 8, u"Эпоха")
        ws.write(0, 9, u"Всего эпох")
        ws.write(0, 10, u"Ширина")
        ws.write(0, 11, u"Высота")
        ws.write(0, 12, u"Обучающая выборка")
        ws.write(0, 13, u"Валидационная выборка")
        ws.write(0, 14, u"Время обучения")
        ws.write(0, 15, u"Batch size")
        ws.write(0, 16, u"Accuracy - test")
        ws.write(0, 17, u"Loss - test")
        ws.write(0, 18, u"Recall - test")
        ws.write(0, 19, u"Precision - test")
        ws.write(0, 20, u"F-мера - test")
        ws.write(0, 21, u"Время теста")
        ws.write(0, 22, u"Кол-во тестов")
        i = 1
        for ns in list:
            print (ns)
            model.load_weights(conf.ns + '/' + ns)
            model.compile(loss='binary_crossentropy',
                          optimizer='rmsprop',
                          metrics=['accuracy',
                                   precision_threshold(0.5),
                                   recall_threshold(0.5)])
            out_json = test(model=model, conf=conf)
            data = {}
            file = ns.split('.')
            f = file[0] + '.json'
            print (f)
            with open(conf.ns +'/'+ f) as data_file:
                data = json.load(data_file, encoding='utf-8')
            ws.write(i, 0, f)
            ws.write(i, 1, data['loos_end'])
            ws.write(i, 2, data['acc_end'])
            ws.write(i, 3, data['val_loss'])
            ws.write(i, 4, data['val_acc'])
            try:
                ws.write(i, 5, data['val_recall'])
            except:
                ws.write(i, 5, 0)
            try:
                ws.write(i, 6, data['val_precision'])
            except:
                ws.write(i, 6, 0)
            try:
                ws.write(i, 7, data['f1_measure'])
            except:
                ws.write(i, 7, 0)
            ws.write(i, 8, data['ep_time'])
            ws.write(i, 9, data['epochs'])
            ws.write(i, 10, data['img_width'])
            ws.write(i, 11, data['img_height'])
            ws.write(i, 12, data['train_samples'])
            ws.write(i, 13, data['validation_samples'])
            ws.write(i, 14, data['time_sec'])
            ws.write(i, 15, data['batch_size'])
            ws.write(i, 16, out_json.loss)
            ws.write(i, 17, out_json.accuracy)
            ws.write(i, 18, out_json.precision)
            ws.write(i, 19, out_json.recall)
            ws.write(i, 20, out_json.f1_measure)
            ws.write(i, 21, out_json.timeSec)
            ws.write(i, 22, conf.testCount)
            i += 1
        w.save(conf.ns + '/test_res.xls')
        pass  # отправная точка приложения.


if __name__ == '__main__':
    arg = Dynamic()
    arg.isTrain = False
    main(arg)
