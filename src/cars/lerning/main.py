# -*- coding: utf-8 -*-
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import Callback
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
import time


class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.acc = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.acc.append(logs.get('acc'))


# dimensions of our images.
img_width, img_height = 150, 150

train_data_dir = 'C:/aknysh/carsTrain/forwardBack/train'
validation_data_dir = 'C:/aknysh/carsTrain/forwardBack/validation'
nb_train_samples = 17540
nb_validation_samples = 3130
epochs = 100
batch_size = 10
weights_path = 'C:/aknysh/carsTrain/forwardBack/res/forwardBack'

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
model.add(Dense(60))
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
ep = 1
h = object

while ep <= epochs:
    history = LossHistory()
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

    f = open(weights_path + '_' + str(ep) + '.txt', 'w')
    # вывод в фаил
    f.write('Loss:\n')
    for item in history.losses:
        f.write(str(item) + '\n')

    f.write('Acc:\n')
    for item in history.acc:
        f.write(str(item) + '\n')

    f.write('time:  {:.6} \n'.format(endTime - startTime))
    f.close()

    print('ConvNet is trained\n')
    model.save_weights(weights_path + '_' + str(ep) + '.h5')

    print('ConvNet is saved\n')
    ep += 1

print('---end---\n')
