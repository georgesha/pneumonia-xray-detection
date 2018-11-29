# read chest x-ray images
import os
import cv2
import numpy as np

# read images in normal and pneumonia folders


def readImages(path):
    imagesNormal = []
    imagesPneumonia = []
    count_n = 0
    count_p = 0
    for filename in os.listdir(path + "/NORMAL/"):
        count_n = count_n + 1
        # if count_n == 1000:
        #     break
        if filename.split(".")[-1] == "jpeg":
            image = cv2.imread(path + "/NORMAL/" + filename, 0)
            image = cv2.resize(image, (256, 256))
            x, y = image.shape
            image = np.reshape(image, x * y)
            imagesNormal.append(image)
    for filename in os.listdir(path + "/PNEUMONIA/"):
        count_p = count_n + 1
        # if count_p == 1000:
        #     break
        if filename.split(".")[-1] == "jpeg":
            image = cv2.imread(path + "/PNEUMONIA/" + filename, 0)
            image = cv2.resize(image, (256, 256))
            x, y = image.shape
            image = np.reshape(image, x * y)
            imagesPneumonia.append(image)

    return imagesNormal, imagesPneumonia
