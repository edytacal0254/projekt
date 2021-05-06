#import csv
#import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os


images_dir = os.getcwd() + "\\e\\train2017"

images_list = os.listdir(images_dir)

# YOLO settings
labelsPath = os.getcwd() + "\\yolo-coco\\coco.names"
LABELS = open(labelsPath).read().strip().split("\n")

weightsPath = os.getcwd() + "\\yolo-coco\\yolov3.weights"
configPath = os.getcwd() + "\\yolo-coco\\yolov3.cfg"

print("Loading YOLO")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

# initialize a list of colors to represent each possible class label
# if gen_images:
#     np.random.seed(42)
#     COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

desired_confidence = 0.5
desired_threshold = 0.3

nr_of_images = str(len(images_list))
# detecting
used = 0
correct_size = 0
not_interesting = 0


for idx_name, img_name in enumerate(images_list):

    if idx_name == 100:
        print("used: ", used)
        break

    if ((idx_name + 1) % 100) == 0:
        print("Processed : [ " + str(idx_name + 1) + " : " + nr_of_images + " ]")

    image = cv2.imread(images_dir + "\\" + img_name)
    (H, W) = image.shape[:2]

    if (H == 1080) or (W == 1920):
        correct_size += 1
        continue

print("correct : ", correct_size)