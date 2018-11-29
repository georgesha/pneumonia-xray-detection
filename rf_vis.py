# Visualize random forest model
import pickle
from sklearn.tree import export_graphviz
from sklearn.metrics import confusion_matrix, f1_score, roc_curve, auc
import cv2
import numpy as np
import matplotlib.pyplot as plt

import image

test_n, test_p = image.readImages("E:/Machine Learning/chest_xray/test")
test = test_n + test_p
print("Read test data done")

# Create test labels
test_labels = []
test_labels += len(test_n) * [0]
test_labels += len(test_p) * [1]

# # save a single tree
# rf_tree = rf.estimators_[10]
# export_graphviz(rf_tree, out_file='tree.dot', rounded=True,
#                 proportion=False, precision=2, filled=True)

# load model
rf = pickle.load(open('rf.sav', 'rb'))

# extract feature importance
feature_im = np.array(rf.feature_importances_)
top_pixels = feature_im.argsort()[-5000:][::-1]
second_pixels = feature_im.argsort()[-4000:-2001][::-1]
third_pixels = feature_im.argsort()[-6000:-4001][::-1]

image = cv2.imread(
    "E:/Machine Learning/chest_xray/train/PNEUMONIA/person51_bacteria_242.jpeg", 0)
image = cv2.resize(image, (256, 256))
x, y = image.shape
rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

for pixel in top_pixels:
    x = pixel // 256
    y = pixel % 256
    rgb[x, y] = (0, 255, 0)
# for pixel in second_pixels:
#     x = pixel // 256
#     y = pixel % 256
#     rgb[x, y] = (255, 0, 0)
# for pixel in third_pixels:
#     x = pixel // 256
#     y = pixel % 256
#     rgb[x, y] = (0, 0, 255)


cv2.imwrite("rf_feature_importance.png", rgb)


# score and confusion matrix
def draw_confusion_matrix(confusion_matrix, title, filename):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(confusion_matrix)
    plt.title(title)
    fig.colorbar(cax)
    # ax.set_xticks(["Normal", "Pneumonia"])
    # ax.set_yticks(["Normal", "Pneumonia"])
    plt.xlabel("Predicted labels")
    plt.ylabel("True labels")
    plt.savefig(filename)
    plt.clf()

predict_labels = rf.predict(test)
score = rf.score(test, test_labels)
print("Score: " + str(score))

f1_score = f1_score(test_labels, predict_labels)
print("F1 Score: " + str(f1_score))
# Plot confusion matrix
cm = confusion_matrix(test_labels, predict_labels)
draw_confusion_matrix(cm, "Confusion matrix of Random Forest",
                      "rf_confusion_matrix.png")

probs = rf.predict_proba(test)
preds = probs[:,1]
fpr, tpr, threshold = roc_curve(test_labels, preds)
auc = auc(fpr, tpr)
print(auc)
plt.title('Random Forest ROC')
plt.plot(fpr,tpr, color="blue", label="AUC for Random Forest: " + str(auc))
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.savefig("rf_roc.png")
