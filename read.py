#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 24 11:14:42 2018

@author: Zijing
"""


# read chest x-ray images
import os
import cv2
import numpy as np

# labeling 
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

data = np.concatenate((imagesNormal, imagesPneumonia), axis=0)


img_train, labels_train = data.load_training()
train_img = np.array(img_train)
train_labels = np.array(labels_train)