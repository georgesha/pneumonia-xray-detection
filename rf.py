import numpy as np
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier

import image

# Read train data
train_n, train_p = image.readImages("E:/Machine Learning/chest_xray/train")
train = train_n + train_p
print("Read train data done")

# Create train labels
train_labels = []
train_labels += len(train_n) * [0]
train_labels += len(train_p) * [1]

# Read test data
test_n, test_p = image.readImages("E:/Machine Learning/chest_xray/test")
test = test_n + test_p
print("Read test data done")

# Create test labels
test_labels = []
test_labels += len(test_n) * [0]
test_labels += len(test_p) * [1]

f = RandomForestClassifier(n_estimators = 20)
f = f.fit(train, train_labels)

result = f.predict(test)
# for item in result:
#     print(item)
#
score = f.score(test, test_labels)
print("Score: " + str(score))

f1 = metrics.f1_score(test_labels, result)
print("F1: " + str(f1))
