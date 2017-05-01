# -*- coding: utf-8 -*-
# <autor>Кныш Александр</autor>
import io
from keras.preprocessing.image import ImageDataGenerator
import time
import json

from src.cars.lerning.module.Dynamic.Dynamic import Dynamic


def test(model, conf):
    '''
    Метод тестирования обученных нс
    PS: тестирование средствами keras 
    :return: 
    '''
    # model.load_weights(conf.ns)
    # model.compile(loss='binary_crossentropy',
    #               optimizer='rmsprop',
    #               metrics=['accuracy'])
    print('Model is compiled\n')
    print (model.metrics_names)
    little_datagen = ImageDataGenerator(rescale=1. / 255)
    little_generator = little_datagen.flow_from_directory(
        conf.path_test,
        target_size=(conf.img_width, conf.img_height),
        batch_size=1)

    validation_generator = little_datagen.flow_from_directory(
        conf.validation_data_dir,
        target_size=(conf.img_width, conf.img_height),
        batch_size=conf.batch_size,
        class_mode='binary')

    # save_to_dir='/home/aknysh/git/DiplomVision/src/cars/lerning/cars/test',
    start = time.time()
    out = model.predict_generator(validation_generator, 10)
    stop = time.time()
    test = model.evaluate_generator(validation_generator, conf.testCount, workers=100)
    out_json = Dynamic()
    out_json.loss = test[0]
    out_json.accuracy = test[1]
    out_json.precision = test[2]
    out_json.recall = test[3]
    out_json.f1_measure = (2 * out_json.precision * out_json.recall) / (out_json.precision + out_json.recall)
    sec = stop - start
    out_json.timeSec = sec

    print("predict time = %.4f sec" % sec)
    x = vars(out_json)
    with io.open('test.' + '.json', 'w') as f:
        f.write(unicode(json.dumps(x, ensure_ascii=False)))
        # print(out)
