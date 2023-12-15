import os
import cv2

data_directory = ""

categories = []

for category in categories:  # do dogs and cats
    path = os.path.join(data_directory, category)  # create path to dogs and cats
    for img in os.listdir(path):  # iterate over each image per dogs and cats
        img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)