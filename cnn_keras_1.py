import numpy as np
from PIL import Image
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

classifier = Sequential()

classifier.add(Conv2D(32, (6, 6), input_shape=(256, 256, 3), activation='relu'))

classifier.add(MaxPooling2D(pool_size=(3, 3)))

classifier.add(Flatten())

classifier.add(Dense(units=128, activation='relu'))

classifier.add(Dense(units=1, activation='sigmoid'))

classifier.compile(
    optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1. / 255)
training_set = train_datagen.flow_from_directory('./chest_xray/train',
                                                 target_size=(256, 256),
                                                 batch_size=32,
                                                 class_mode='binary')
test_set = test_datagen.flow_from_directory('./chest_xray/test',
                                            target_size=(256, 256),
                                            batch_size=32,
                                            class_mode='binary')

classifier.fit_generator(training_set,
                         steps_per_epoch=8000,
                         epochs=5,
                         validation_data=test_set,
                         validation_steps=1000)
