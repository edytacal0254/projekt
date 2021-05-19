#import csv
#import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
#import math


def add_matrix(mat, x_, y_, w_, h_):
    (A, B) = mat.shape
    h__ = y_ + h_
    w__ = x_ + w_
    if (y_+h_) > A:
        h__ = A
    if (x_+w_) > B:
        w__ = B

    for a in range(y_, h__):
        for b in range(x_, w__):
            mat[a, b] += 1
    return mat

#**********************************************************
#which_one = "apple"
#which_one = "car"
#which_one = "dog"
which_one = "chair"
#**********************************************************

images_dir = os.getcwd() + "\\from_coco_db\\" + which_one
images_list = os.listdir(images_dir)

my_matrix = np.zeros(shape=(480, 640))


# YOLO settings
labelsPath = os.getcwd() + "\\yolo-coco\\coco.names"
LABELS = open(labelsPath).read().strip().split("\n")

weightsPath = os.getcwd() + "\\yolo-coco\\yolov3.weights"
configPath = os.getcwd() + "\\yolo-coco\\yolov3.cfg"

print("Loading YOLO")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)


desired_confidence = 0.5
desired_threshold = 0.3

nr_of_images = str(len(images_list))
# detecting

not_interesting = 0
px = 0
py = 0
orientation = 0
for idx_name, img_name in enumerate(images_list):

    if ((idx_name + 1) % 100) == 0:
        print("Processed : [ " + str(idx_name + 1) + " : " + nr_of_images + " ]")

    image = cv2.imread(images_dir + "\\" + img_name)
    (H, W) = image.shape[:2]

    px = 480/H
    py = 640/W


    # determine only the *output* layer names that we need from YOLO
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    # construct a blob from the input image and then perform a forward
    # pass of the YOLO object detector, giving us our bounding boxes and
    # associated probabilities
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(ln)

    # initialize our lists of detected bounding boxes, confidences, and
    # class IDs, respectively
    boxes = []
    confidences = []
    classIDs = []

    # loop over each of the layer outputs
    for output in layerOutputs:
        # loop over each of the detections
        for detection in output:
            # extract the class ID and confidence (i.e., probability) of
            # the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            # filter out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            if confidence > desired_confidence:
                # scale the bounding box coordinates back relative to the
                # size of the image, keeping in mind that YOLO actually
                # returns the center (x, y)-coordinates of the bounding
                # box followed by the boxes' width and height
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                # use the center (x, y)-coordinates to derive the top and
                # and left corner
                #
                # of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # update our list of bounding box coordinates, confidences,
                # and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    # apply non-maxima suppression to suppress weak, overlapping bounding
    # boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, desired_confidence, desired_threshold)

    # ensure at least one detection exists
    if len(idxs) > 0:
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            # take only data of our chosen classes

            #/////////////////////////////////////////////////////
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            n_x = int(px * x)
            n_y = int(py * y)
            n_w = int(px * w)
            n_h = int(py * h)
            if LABELS[classIDs[i]] == which_one:
                add_matrix(my_matrix, n_x, n_y, n_w, n_h)

print("Processed all " + nr_of_images + " images : " + which_one)
print("not interesting detected objects: ", not_interesting)

plt.imshow(my_matrix, cmap='hot', interpolation='nearest')
plt.savefig(os.getcwd() + "//from_coco_db//heatmaps//" + which_one+".jpg")
plt.show()


print("saved heatmap : " + which_one)
