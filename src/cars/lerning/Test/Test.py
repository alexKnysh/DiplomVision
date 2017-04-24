# -*- coding: utf-8 -*-
# <autor>Кныш Александр</autor>

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import time


def reshape(img):
    print 123
    pass


def test(model, conf):
    '''
    Метод тестирования обученных нс
    PS: тестирование средствами keras 
    :return: 
    '''
    model.load_weights(conf.ns)
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    print('Model is compiled\n')
    img = load_img('/home/aknysh/git/DiplomVision/src/cars/lerning/data/test/06403101721.bmp',
                   target_size=(conf.img_width, conf.img_height))  # this is a PIL image
    # img = load_img('D:/Datasets/dogs_and_cats/data/train/dogs/dog.138.jpg', target_size=(img_width, img_height))
    x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
    x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)
    sample_datagen = ImageDataGenerator(rescale=1. / 255)
    sample_gen = sample_datagen.flow(x, batch_size=1)

    start = time.time()
    out = model.predict_generator(sample_gen, 1)
    stop = time.time()
    sec = stop - start
    print("predict time = %.4f sec" % sec, '\tout =', out[0][0])
    if (out < 0.5):
        print('\tIt is a bad')
    else:
        print('\tIt is a good')
    pass

    # this is the augmentation configuration we will use for testing:
    # only rescaling
    little_datagen = ImageDataGenerator(rescale=1. / 255)
    little_generator = little_datagen.flow_from_directory(
        conf.path_test,
        target_size=(conf.img_width, conf.img_height),
        batch_size=1,
        class_mode=None,
        seed=1,
        shuffle=False)

    start = time.time()
    out = model.predict_generator(little_generator, 10)
    stop = time.time()
    sec = stop - start
    print("predict time = %.4f sec" % sec)
    print(out)

'''
    # img = load_img('D:/Datasets/dogs_and_cats/data/train/cats/cat.63.jpg',target_size=(img_width, img_height))  # this is a PIL image
    img = load_img('D:/Datasets/dogs_and_cats/data/train/dogs/dog.138.jpg', target_size=(conf.img_width, conf.img_height))
    x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
    x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)
    sample_datagen = ImageDataGenerator(rescale=1. / 255)
    sample_gen = sample_datagen.flow(x, batch_size=1)

    start = time.time()
    out = model.predict_generator(sample_gen, 1)
    stop = time.time()
    sec = stop - start
    print("predict time = %.4f sec" % sec, '\tout =', out[0][0], end=' ')
    if (out < 0.5):
        print('\tIt is a cat')
    else:
        print('\tIt is a dog')
'''
