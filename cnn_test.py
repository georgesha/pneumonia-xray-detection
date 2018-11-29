# load cnn and test on test image set
import numpy as np
import matplotlib.pyplot as plt
from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import roc_curve, auc

# load model
file = open('cnn3.json', 'r')
cnn_json = file.read()
file.close()
cnn = model_from_json(cnn_json)
# load weights
cnn.load_weights("cnn3.h5")
print("Loaded model")

cnn.compile(
    optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

test_datagen = ImageDataGenerator(rescale=1. / 255)
test_set = test_datagen.flow_from_directory('E:/Machine Learning/chest_xray/test',
                                            target_size=(256, 256),
                                            batch_size=32,
                                            class_mode='binary')

score = cnn.evaluate_generator(test_set)
print(score)
exit()

test_labels = test_set.classes[test_set.index_array]
test_predict = cnn.predict_generator(test_set).ravel()
preds = []
for i in range(len(test_predict)):
    if test_labels[i] == 1:
        preds.append(test_predict[i])
    else:
        preds.append(1 - test_predict[i])
fpr, tpr, threshold = roc_curve(test_labels, preds)
auc = auc(fpr, tpr)
print(auc)
plt.title('CNN ROC')
plt.plot(fpr,tpr,color="red")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.savefig("cnn_roc.png")
