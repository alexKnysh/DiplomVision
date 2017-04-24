# -*- coding: utf-8 -*-
# <autor>Кныш Александр</autor>

import time
import json, io
from keras import backend as K
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from src.cars.lerning.module import LossHistory
from outJsonData import OutJsonData


def train():
    # размеры изображений.
    img_width, img_height = 36, 36

    # пути к обучающим и валидационным сетам
    train_data_dir = '/home/aknysh/git/DiplomVision/src/cars/lerning/data/train'
    validation_data_dir = '/home/aknysh/git/DiplomVision/src/cars/lerning/data/validation'
    #  выходные сети (по эпохам)
    weights_path = '/home/aknysh/git/DiplomVision/src/cars/lerning/res/forwardBack'
    nb_train_samples = 10600  # количестко самплев обучения
    nb_validation_samples = 4680  # количестко самплев валидации
    epochs = 100  # кол-во всего эпох
    batch_size = 10
    ep = 1  # c какой эпохи начинать обучение...
    if K.image_data_format() == 'channels_first':
        input_shape = (3, img_width, img_height)
    else:
        input_shape = (img_width, img_height, 3)

    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(120))  # TODO: было 60!!!!!!!
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    print('Model is compiled\n')

    # this is the augmentation configuration we will use for training
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    # this is the augmentation configuration we will use for testing:
    # only rescaling
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary')

    validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary')

    print('Data is generated from folders\n')

    h = object

    while ep <= epochs:
        history = LossHistory.LossHistory()
    startTime = time.time()
    if (ep - 1 > 0):
        model.load_weights(weights_path + '_' + str(ep - 1) + '.h5')
        model.fit_generator(
            train_generator,
            callbacks=[history],
            steps_per_epoch=nb_train_samples // batch_size,
            epochs=ep,
            validation_data=validation_generator,
            validation_steps=nb_validation_samples // batch_size,
            initial_epoch=ep - 1)
    else:
        model.fit_generator(
            train_generator,
            callbacks=[history],
            steps_per_epoch=nb_train_samples // batch_size,
            epochs=ep,
            validation_data=validation_generator,
            validation_steps=nb_validation_samples // batch_size)
    endTime = time.time()

    # вывод в фаил
    outJsonData = OutJsonData(acc=history.acc,
                              loss=history.losses,
                              train_samples=nb_train_samples,
                              validation_samples=nb_validation_samples,
                              batch_size=batch_size,
                              img_width=img_width,
                              img_height=img_height,
                              epochs=epochs,
                              ep_time=ep,
                              time=endTime - startTime)
    x = vars(outJsonData)
    with io.open(weights_path + '_' + str(ep) + '.json', 'w') as f:
        f.write(unicode(json.dumps(x, ensure_ascii=False)))

    print('ConvNet is trained\n')
    model.save_weights(weights_path + '_' + str(ep) + '.h5')

    print('ConvNet is saved\n')
    ep += 1
