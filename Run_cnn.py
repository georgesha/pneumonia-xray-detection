#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 19:15:17 2018

@author: MOOSE
"""

# Importing the Keras libraries and packages
import os ; os.environ['HDF5_DISABLE_VERSION_CHECK']='2'
import numpy as np
import tensorflow
import pandas as pd


from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))
"""
training images. Since we are working on images here, 
which a basically 2 Dimensional arrays, we’re using Convolution 2-D

Conv2D (number of filters, shape of the filter, input shape and 3 since RBG. 
Activation is relu for rectifer function
"""
# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))
"""
MaxPooling we need the maximum value pixel from the respective region of interest
"""
# Adding a second convolutional layer
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

"""
Pooling layer reduces the size of the images as much as possible.
aka, reduced the complexity of the model without reducing it’s performanc

we’ll have minimum pixel loss and get a precise region where the feature are located.
"""

# Step 3 - Flattening
classifier.add(Flatten())

"""
Flatten from keras.layers, which is used for Flattening. 
Flattening is the process of converting all the result into 2 dimensional 
arrays into a single long continuous linear vector.
"""

# Step 4 - Full connection
classifier.add(Dense(units = 64, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))

"""
Dense is the function to add a fully connected layer, ‘units’ is where we 
define the number of nodes that should be present in this hidden layer, 
these units value will be always between the number of input nodes and 
the output nodes but the art of choosing the most optimal number of nodes 
can be achieved only through experimental tries. Though it’s a common practice
to use a power of 2. And the activation function will be a rectifier function.
 
Finallu, our output layer, which should contain only one node, as it is binary classification
"""
# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
#classifier.save(“weights.h5”)

# Part 2 - Fitting the CNN to the images
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1./255,
shear_range = 0.2,
zoom_range = 0.2,
horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255)
training_set = train_datagen.flow_from_directory('dataset/train',
target_size = (64, 64),
batch_size = 32,
class_mode = 'binary')
test_set = test_datagen.flow_from_directory('dataset/test',
target_size = (64, 64),
batch_size = 32,
class_mode = 'binary')
classifier.fit_generator(training_set,
steps_per_epoch = 100,
epochs = 5,
validation_data = test_set,
validation_steps = 20)
"""
training set number of files, test set number of files

1 Epoch is a single step that trains a neural network.
in this case, we train it 25 times.

"""
# Part 3 - Making new predictions
import numpy as np
from keras.preprocessing import image
test_image = image.load_img('dataset/single_prediction/person1949_bacteria_4880.jpeg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
training_set.class_indices
if result[0][0] == 1:
    prediction = 'pnenomia'
else:
    prediction = 'normal'

