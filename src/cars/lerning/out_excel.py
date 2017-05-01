# -*- coding: utf-8 -*-
# <autor>Кныш Александр</autor>

import json
import os
from uu import decode

from xlwt import *


def create_log(conf):
    p = conf.weights_path.split('/')
    st = '/'.join(p[0: p.__len__() - 2])
    list_dir = os.listdir(st)
    for dir in list_dir:
        i = 1
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
        list = os.listdir(st + '/' + dir)
        list1 = filter(lambda x: x.endswith('json'), list)
        list_file = filter(lambda x: x.find('model') < 0, list1)
        for file in list_file:
            data = {}
            print (file)
            with open(st + '/' + dir + '/' + file) as data_file:
                data = json.load(data_file, encoding='utf-8')
            ws.write(i, 0, file)
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
            i += 1
        w.save(st + '/' + dir + '/learning_res.xls')
