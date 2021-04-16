import csv
import numpy as np
import cv2
import os

#
# generate and show images after detection (no saving)
# gen_images = False
#
#

# input images and csv file
# \\created\\generated
images_dir = os.getcwd() + "\\created\\generated"
data_gen_path = os.getcwd() + "\\created\\data_gen.csv"
if not (os.path.isdir(images_dir) and os.path.isfile(data_gen_path)):
    print("No or incomplete to analise")
    exit()

images_list = os.listdir(images_dir)
detecting_classes = ["apple", "dog", "car", "chair"]

data_det_path = os.getcwd() + "\\created\\data_det.csv"
data_det_file = open(data_det_path, "w", encoding='utf-8', newline='')
data_det_writer = csv.writer(data_det_file)

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
for idx_name, img_name in enumerate(images_list):
    # if idx_name == 100:
    #    break
    if ((idx_name + 1) % 100) == 0:
        print("Processed : [ " + str(idx_name + 1) + " : " + nr_of_images + " ]")

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
                # if gen_images:
                #     color = [int(c) for c in COLORS[classIDs[i]]]
                #     cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                #     text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
                #     cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # fill csv file
                row = [img_name, LABELS[classIDs[i]], confidences[i], x, y, w, h]
                data_det_writer.writerow(row)
                # print(LABELS[classIDs[i]], img_name)

    # show the output image
    # if gen_images:
    #     cv2.imshow("Image", image)
    #     cv2.waitKey(0)

print("Processed all " + nr_of_images + " images")
