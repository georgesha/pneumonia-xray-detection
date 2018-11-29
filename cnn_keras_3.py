import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

classifier = Sequential()

classifier.add(Conv2D(32, (6, 6), input_shape=(256, 256, 3), activation='relu'))

classifier.add(MaxPooling2D(pool_size=(3, 3)))

classifier.add(Conv2D(64, (3, 3), activation='relu'))

classifier.add(MaxPooling2D(pool_size=(2, 2)))

classifier.add(Conv2D(128, (3, 3), activation='relu'))

classifier.add(MaxPooling2D(pool_size=(2, 2)))

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
training_set = train_datagen.flow_from_directory('E:/Machine Learning/chest_xray/train',
                                                 target_size=(256, 256),
                                                 batch_size=32,
                                                 class_mode='binary')
val_set = test_datagen.flow_from_directory('E:/Machine Learning/chest_xray/val',
                                            target_size=(256, 256),
                                            batch_size=32,
                                            class_mode='binary')

history = classifier.fit_generator(training_set,
                         steps_per_epoch=1000,
                         epochs=10,
                         validation_data=val_set,
                         validation_steps=1000)

print(history.history['acc'])

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Accuracy')
plt.ylabel('accuracy')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig("accuracy.png")
plt.clf()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Loss')
plt.ylabel('loss')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig("loss.png")

# score = classifier.evaluate(test_set)
# print("Score: " + str(score))
#
classifier_json = classifier.to_json()
with open("cnn3_10.json", "w") as json_file:
    json_file.write(classifier_json)
classifier.save_weights("cnn3_10.h5")
print("Saved model")
