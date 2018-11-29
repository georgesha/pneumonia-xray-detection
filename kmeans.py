from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

import image

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

pca = PCA(n_components=2)
pca.fit(train)
train_pca = pca.transform(train)
test_pca = pca.transform(test)
print("PCA done")

kmeans = KMeans(n_clusters=2, random_state=0, n_jobs=-1)
kmeans.fit(train_pca)
pred_labels = kmeans.predict(test_pca)
plt.scatter(test_pca[:, 0], test_pca[:, 1], c=pred_labels, cmap='viridis')
plt.savefig("kmeans.png")
