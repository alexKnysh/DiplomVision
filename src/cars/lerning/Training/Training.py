# -*- coding: utf-8 -*-
# <autor>Кныш Александр</autor>

import time
import json, io
from keras.preprocessing.image import ImageDataGenerator
from src.cars.lerning.module import LossHistory
from outJsonData import OutJsonData


def train(model, conf):
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
        conf.train_data_dir,
        target_size=(conf.img_width, conf.img_height),
        batch_size=conf.batch_size,
        class_mode='binary')

    validation_generator = test_datagen.flow_from_directory(
        conf.validation_data_dir,
        target_size=(conf.img_width, conf.img_height),
        batch_size=conf.batch_size,
        class_mode='binary')

    print('Data is generated from folders\n')

    while conf.ep <= conf.epochs:
        history = LossHistory.LossHistory()
        # test = TestCallback()
        startTime = time.time()
        if (conf.ep - 1 > 0):
            model.load_weights(conf.weights_path + '_' + str(conf.ep - 1) + '.h5')
            model.fit_generator(
                train_generator,
                callbacks=[history],
                steps_per_epoch=conf.nb_train_samples // conf.batch_size,
                epochs=conf.ep,
                validation_data=validation_generator,
                validation_steps=conf.nb_validation_samples // conf.batch_size,
                initial_epoch=conf.ep - 1)
        else:
            model.fit_generator(
                train_generator,
                callbacks=[history],
                steps_per_epoch=conf.nb_train_samples // conf.batch_size,
                epochs=conf.ep,
                validation_data=validation_generator,
                validation_steps=conf.nb_validation_samples // conf.batch_size)
        endTime = time.time()

        scoreSeg = model.evaluate_generator(validation_generator, conf.nb_validation_samples)

        # вывод в фаил
        outJsonData = OutJsonData(acc=history.acc,
                                  loss=history.losses,
                                  train_samples=conf.nb_train_samples,
                                  validation_samples=conf.nb_validation_samples,
                                  batch_size=conf.batch_size,
                                  img_width=conf.img_width,
                                  img_height=conf.img_height,
                                  epochs=conf.epochs,
                                  ep_time=conf.ep,
                                  time=endTime - startTime)
        x = vars(outJsonData)
        with io.open(conf.weights_path + '_' + str(conf.ep) + '.json', 'w') as f:
            f.write(unicode(json.dumps(x, ensure_ascii=False)))

        print('ConvNet is trained\n')
        model.save_weights(conf.weights_path + '_' + str(conf.ep) + '.h5')

        print('ConvNet is saved\n')

        conf.ep += 1
