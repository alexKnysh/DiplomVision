# -*- coding: utf-8 -*-
# <autor>Кныш Александр</autor>
import io
import json

from keras import backend as K
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential


def precision_threshold(threshold=0.5):
    '''
    Вычисление точности обнаружения
    :param threshold: Порог срабатывания
    :return:  Метод для вычисления точности (Precision) детектирования.
    '''

    def precision(y_true, y_pred):
        """Precision metric.
        Computes the precision over the whole batch using threshold_value.
        """
        threshold_value = threshold
        # Adaptation of the "round()" used before to get the predictions. Clipping to make sure that the predicted raw values are between 0 and 1.
        y_pred = K.cast(K.greater(K.clip(y_pred, 0, 1), threshold_value), K.floatx())
        # Compute the number of true positives. Rounding in prevention to make sure we have an integer.
        true_positives = K.round(K.sum(K.clip(y_true * y_pred, 0, 1)))
        # count the predicted positives
        predicted_positives = K.sum(y_pred)
        # Get the precision ratio
        precision_ratio = true_positives / (predicted_positives + K.epsilon())
        return precision_ratio

    return precision


def recall_threshold(threshold=0.5):
    '''
    Вычисление полноты детектирования
    :param threshold: порог детектирования
    :return:  Метод для вычисления полноты (Recall) детектирования.
    '''

    def recall(y_true, y_pred):
        """Recall metric.
        Computes the recall over the whole batch using threshold_value.
        """
        threshold_value = threshold
        # Adaptation of the "round()" used before to get the predictions. Clipping to make sure that the predicted raw values are between 0 and 1.
        y_pred = K.cast(K.greater(K.clip(y_pred, 0, 1), threshold_value), K.floatx())
        # Compute the number of true positives. Rounding in prevention to make sure we have an integer.
        true_positives = K.round(K.sum(K.clip(y_true * y_pred, 0, 1)))
        # Compute the number of positive targets.
        possible_positives = K.sum(K.clip(y_true, 0, 1))
        recall_ratio = true_positives / (possible_positives + K.epsilon())
        return recall_ratio

    return recall


def create_model(img_width=36, img_height=36, path_save=''):
    '''
    Создание архитектуры сети.
    :param img_width: Ширина изображения.
    :param img_height: Высота изображения. 
    :return: модель сети для обучения.
    '''
    if K.image_data_format() == 'channels_first':
        input_shape = (3, img_width, img_height)
    else:
        input_shape = (img_width, img_height, 3)

    model = Sequential()
    model.add(Conv2D(128, (3, 3), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(256, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(100))  # TODO: Было 60!!!!!!!
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy', precision_threshold(0.5), recall_threshold(0.5)])
    print('Model is compiled\n')
    json_string = model.to_json()
    with io.open(path_save + '_model' + '.json', 'w') as f:
        f.write(unicode(json.dumps(json_string, ensure_ascii=False)))
    print ("Model is save..." + path_save + '_model' + '.json')

    return model
