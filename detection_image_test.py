import csv
import numpy as np
import cv2
import os

#
# generate and show images after detection (no saving)
gen_images = True
#
#

# input images and csv file
# \\created\\generated
images_dir = os.getcwd() + "\\created\\_v1\\generated_v1"


#images_list = ["B11_003796.jpg"]
#images_list = ["B14_004126.jpg","B14_004128.jpg","B14_004129.jpg"] #not ok
#images_list = ["B14_004125.jpg","B14_004127.jpg"] #ok
#images_list = ["D06_008238.jpg", "D06_008241.jpg", "D06_008243.jpg", "D06_008244.jpg", "D07_008363.jpg"]
images_list = ["C08_005884.jpg", "C08_005914.jpg"]
detecting_classes = ["apple", "dog", "car", "chair"]


# YOLO settings
labelsPath = os.getcwd() + "\\yolo-coco\\coco.names"
LABELS = open(labelsPath).read().strip().split("\n")

weightsPath = os.getcwd() + "\\yolo-coco\\yolov3.weights"
configPath = os.getcwd() + "\\yolo-coco\\yolov3.cfg"

print("Loading YOLO")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

# initialize a list of colors to represent each possible class label
if gen_images:
    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

desired_confidence = 0.5
desired_threshold = 0.3


for idx_name, img_name in enumerate(images_list):

    image = cv2.imread(images_dir + "\\" + img_name)
    (H, W) = image.shape[:2]

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
            if LABELS[classIDs[i]] in detecting_classes:
                # extract the bounding box coordinates
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])

                # draw a bounding box rectangle and label on the image
                if gen_images:
                    color = [int(c) for c in COLORS[classIDs[i]]]
                    cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                    text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
                    cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # fill csv file

                print(LABELS[classIDs[i]], img_name)

    # show the output image
    if gen_images:
        filename = os.getcwd() + "\\created\\" + img_name
        cv2.imwrite(filename, image)
        cv2.imshow("Image", image)
        cv2.waitKey(0)

print("done")
