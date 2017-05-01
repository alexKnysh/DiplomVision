# -*- coding: utf-8 -*-

class OutJsonData(object):
    '''
    Объект структурированного вывода данных    
    '''

    def __init__(self,
                 loss=[],
                 acc=[],
                 train_samples=0,
                 validation_samples=0,
                 epochs=0,
                 img_width=0,
                 img_height=0,
                 ep_time=0,
                 val_loss=0,
                 val_acc=0,
                 val_recall=0,
                 val_precision=0,
                 batch_size=0,
                 time=0):
        '''
        Конструктор объекта
        :param loss:  значение функции потерь
        :param acc: значение точности
        :param train_samples: кол-во оучающей выборки
        :param validation_samples: кол-во выборки проверок
        :param epochs: кол-во эпох
        :param ep_time: текущая эпоха
        :param batch_size: значения батча
        :param time: время
        '''
        self.loss = loss
        self.acc = acc
        self.time_sec = time
        self.train_samples = train_samples
        self.validation_samples = validation_samples
        self.epochs = epochs
        self.ep_time = ep_time
        self.batch_size = batch_size
        self.img_width = img_width
        self.img_height = img_height
        self.loos_end = self.loss[self.loss.__len__() - 1]
        self.acc_end = self.acc[self.acc.__len__() - 1]
        self.val_loss = val_loss
        self.val_acc = val_acc
        self.val_recall = val_recall
        self.val_precision = val_precision
        self.f1_measure = (2 * self.val_recall * self.val_precision) / (self.val_recall + self.val_precision)