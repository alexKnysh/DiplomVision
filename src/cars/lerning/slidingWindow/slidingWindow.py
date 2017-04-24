# coding: utf8

import cv2
import time
import uuid
from keras.preprocessing.image import ImageDataGenerator, img_to_array


class SlidingWindow:
    '''
    Объект для работы с плавающим окном. (пока cpu)
    '''

    def __init__(self,
                 width=150,
                 height=150,
                 step_x=50,
                 step_y=10,
                 model=None,
                 delta=250):
        '''
        Конструктор объета
        :param width: Ширина скользящего окна.
        :param height: Высота скользящего окна.
        :param step_x: Шаг скользящего окна по x.
        :param step_y: Шаг скользящего окна по y.
        :return: Экземпляр объекта.
        '''
        self.width = width
        self.height = height
        self.step_x = step_x
        self.step_y = step_y
        self.delta = delta
        self.model = model

    def detect_callback(self, detect):
        '''
        Set function detector
        :param detect: detection function
        :return: void.
        '''
        self.detect = detect

    def _solve_first_position(self, _w, w_img):
        '''
        Вычисление положения первого окна.
        :param _w:  параметр объекта.
        :param w_img: параметр изображения.
        :return: одна координата смещения.
        '''
        w = w_img % _w
        if (w == 0):
            return _w
        else:
            return w / 2

    def sliding_windows(self, test_img, ns, img_width, img_height, path_dir):
        '''
        Вычисдение скользящего окна.
        :param img: Изображение для вычисления
        :return: Массив изображений
        '''
        if (not test_img is None):
            h_img = test_img.shape[0]
            w_img = test_img.shape[1]
            is_width = True
            is_heigh = True
            while is_width | is_heigh:
                # определили смещение
                _y_delta = self._solve_first_position(self.height, h_img)
                _x_delta = self._solve_first_position(self.width, w_img)
                y = _y_delta

                while (y + self.height) <= (h_img - _y_delta):
                    x = _x_delta
                    # x,y - текущие координаты

                    while (x + self.width) <= (w_img - _x_delta):
                        _img = test_img.copy()
                        # TODO: Тут будет вызов функции распознавания

                        # self.model.load_weights(ns)
                        # self.model.compile(loss='binary_crossentropy',
                        #                    optimizer='rmsprop',
                        #                    metrics=['accuracy'])
                        img_Roi = _img[y:y+self.height, x:x+self.width ]
                        cv2.imshow('test0', img_Roi)
                        cv2.imwrite(path_dir +'/'+ str(uuid.uuid4())+'.jpg',img_Roi)
                        cv2.rectangle(_img, (x, y), (x + self.width, y + self.height), (255, 7, 7), 2)

                        # x_t = img_to_array(img_Roi)  # this is a Numpy array with shape (3, 150, 150)
                        # x_t = x_t.reshape((1,) + x_t.shape)  # this is a Numpy array with shape (1, 3, 150, 150)
                        # sample_datagen = ImageDataGenerator(rescale=1. / 255)
                        # sample_gen = sample_datagen.flow(x_t, batch_size=1)
                        #
                        # start = time.time()
                        # out = self.model.predict_generator(sample_gen, 1)
                        # stop = time.time()
                        # sec = stop - start
                        # print("predict time = %.4f sec" % sec, '\tout =', out[0][0])
                        # if (out < 0.5):
                        #     print('\tIt is a bad')
                        # else:
                        #     print('\tIt is a good')
                        # pass
                        cv2.imshow('test1', _img)
                        cv2.waitKey(5)
                        x = x + self.step_x
                    y = y + self.step_y

                self.width = self.width + self.delta
                self.height = self.height + self.delta
                is_width = self.width <= w_img
                is_heigh = self.height <= h_img
        else:
            return 0
