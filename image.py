# read chest x-ray images
import os
import cv2
import numpy as np

# read images in normal and pneumonia folders
def readImages(path):
    imagesNormal = []
    imagesPneumonia = []
    for filename in os.listdir(path + "/NORMAL/"):
        if filename.split(".")[-1] == "jpeg":
            image = cv2.imread(path + "/NORMAL/" + filename, 0)
            imagesNormal.append(image)
    for filename in os.listdir(path + "/PNEUMONIA/"):
        if filename.split(".")[-1] == "jpeg":
            image = cv2.imread(path + "/PNEUMONIA/" + filename, 0)
            imagesPneumonia.append(image)

    return imagesNormal, imagesPneumonia
