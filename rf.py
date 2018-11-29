from collections import OrderedDict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
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

# rfs = [
#     ("sqrt",
#         RandomForestClassifier(n_estimators=20,
#                                warm_start=True,
#                                oob_score=True,
#                                max_features="sqrt",
#                                random_state=100,
#                                n_jobs=-1)),
#     ("log2",
#         RandomForestClassifier(n_estimators=20,
#                                warm_start=True,
#                                max_features='log2',
#                                oob_score=True,
#                                random_state=100,
#                                n_jobs=-1))
# ]
#
# error_rate = OrderedDict((label, []) for label, _ in rfs)
#
# for label, rf in rfs:
#     print(label)
#     for i in range(100, 501, 10):
#         print("Trees: " + str(i))
#         rf.set_params(n_estimators=i)
#         rf.fit(train, train_labels)
#
#         # Record the OOB error for each `n_estimators=i` setting.
#         oob_error = 1 - rf.oob_score_
#         error_rate[label].append((i, oob_error))
#
# print("Error rate calculated")
#
# Generate the "OOB error rate" vs. "n_estimators" plot.
# for label, err in error_rate.items():
#     xs, ys = zip(*err)
#     plt.plot(xs, ys, label=label)
#
# plt.xlim(100, 500)
# plt.xlabel("Number of trees")
# plt.ylabel("OOB error")
# plt.legend(loc="upper right")
# plt.savefig('oob.png')

rf = RandomForestClassifier(n_estimators=250, max_features='sqrt', n_jobs=-1)
rf.fit(train, train_labels)

score = rf.score(test, test_labels)
print("Score: " + str(score))

# save model
pickle.dump(rf, open('rf.sav', 'wb'))
