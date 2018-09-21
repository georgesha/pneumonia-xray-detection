# input chest x-ray images
import os
import cv2
import numpy as np

images = []
path = "./images/"
for filename in os.listdir(path):
    image = cv2.imread(path + filename, 0)
    images.append(image)
