#import csv
#import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os


def add_matrix(mat, x_, y_, w_, h_):
    for a in range(y_, h_+1):
        for b in range(x_, w_+1):
            mat[a, b] += 1
    return mat
#
# generate and show images after detection (no saving)
# gen_images = False
#
#


# input images and csv file
# \\created\\generated
images_dir = os.getcwd() + "\\e\\train2017"

#data_gen_path = os.getcwd() + "\\created\\data_gen.csv"
#if not (os.path.isdir(images_dir) and os.path.isfile(data_gen_path)):
#    print("No or incomplete to analise")
#    exit()

images_list = os.listdir(images_dir)
detecting_classes = ["apple", "dog", "car", "chair"]

APPLE_1 = np.zeros(shape=(1080, 1920))
CAR_1 = np.zeros(shape=(1080, 1920))
DOG_1 = np.zeros(shape=(1080, 1920))
CHAIR_1 = np.zeros(shape=(1080, 1920))

APPLE_2 = np.zeros(shape=(1080, 1920))
CAR_2 = np.zeros(shape=(1080, 1920))
DOG_2 = np.zeros(shape=(1080, 1920))
CHAIR_2 = np.zeros(shape=(1080, 1920))

APPLE_3 = np.zeros(shape=(1080, 1920))
CAR_3 = np.zeros(shape=(1080, 1920))
DOG_3 = np.zeros(shape=(1080, 1920))
CHAIR_3 = np.zeros(shape=(1080, 1920))

#data_det_path = os.getcwd() + "\\created\\data_det.csv"
#data_det_file = open(data_det_path, "w", encoding='utf-8', newline='')
#data_det_writer = csv.writer(data_det_file)

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

not_interesting = 0
Hw = 0
hW = 0
hw = 0
px = 0
py = 0
orientation = 0
for idx_name, img_name in enumerate(images_list):

    if idx_name == 60000:
        break

    if ((idx_name + 1) % 100) == 0:
        print("Processed : [ " + str(idx_name + 1) + " : " + nr_of_images + " ]")

    image = cv2.imread(images_dir + "\\" + img_name)
    (H, W) = image.shape[:2]

    if H > W:
        orientation = 1
        Hw += 1
        px = 640/H
        py = 480/W
    elif W > H:
        orientation = 2
        hW += 1
        px = 480/H
        py = 640/W
    else:
        orientation = 3
        hw += 1
        px = 510/H
        py = px

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

            if LABELS[classIDs[i]] == "apple":
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])

                n_x = int(px * x)
                n_y = int(py * y)
                n_w = int(px * w)
                n_h = int(py * h)
                if orientation == 1:
                    add_matrix(APPLE_1, n_x, n_y, n_w, n_h)
                elif orientation == 2:
                    add_matrix(APPLE_2, n_x, n_y, n_w, n_h)
                elif orientation == 3:
                    add_matrix(APPLE_3, n_x, n_y, n_w, n_h)

            elif LABELS[classIDs[i]] == "car":
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])

                n_x = int(px * x)
                n_y = int(py * y)
                n_w = int(px * w)
                n_h = int(py * h)
                if orientation == 1:
                    add_matrix(CAR_1, n_x, n_y, n_w, n_h)
                elif orientation == 2:
                    add_matrix(CAR_2, n_x, n_y, n_w, n_h)
                elif orientation == 3:
                    add_matrix(CAR_3, n_x, n_y, n_w, n_h)

            elif LABELS[classIDs[i]] == "dog":
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])

                n_x = int(px * x)
                n_y = int(py * y)
                n_w = int(px * w)
                n_h = int(py * h)
                if orientation == 1:
                    add_matrix(DOG_1, n_x, n_y, n_w, n_h)
                elif orientation == 2:
                    add_matrix(DOG_2, n_x, n_y, n_w, n_h)
                elif orientation == 3:
                    add_matrix(DOG_3, n_x, n_y, n_w, n_h)

            elif LABELS[classIDs[i]] == "chair":
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])

                n_x = int(px * x)
                n_y = int(py * y)
                n_w = int(px * w)
                n_h = int(py * h)
                if orientation == 1:
                    add_matrix(CHAIR_1, n_x, n_y, n_w, n_h)
                elif orientation == 2:
                    add_matrix(CHAIR_2, n_x, n_y, n_w, n_h)
                elif orientation == 3:
                    add_matrix(CHAIR_3, n_x, n_y, n_w, n_h)

            else:
                not_interesting += 1

            #////////////////////////
            #if LABELS[classIDs[i]] in detecting_classes:
            #    # extract the bounding box coordinates
            #    (x, y) = (boxes[i][0], boxes[i][1])
            #    (w, h) = (boxes[i][2], boxes[i][3])

                # draw a bounding box rectangle and label on the image
                # if gen_images:
                #     color = [int(c) for c in COLORS[classIDs[i]]]
                #     cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                #     text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
                #     cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # fill csv file
                #row = [img_name, LABELS[classIDs[i]], confidences[i], x, y, w, h]
                #data_det_writer.writerow(row)
                # print(LABELS[classIDs[i]], img_name)

    # show the output image
    # if gen_images:
    #     cv2.imshow("Image", image)
    #     cv2.waitKey(0)

print("Processed all " + nr_of_images + " images")
print("not interesting : ", not_interesting)

tmp = [APPLE_1, APPLE_2, APPLE_3, CAR_1, CAR_2, CAR_3, DOG_1, DOG_2, DOG_3, CHAIR_1, CHAIR_2, CHAIR_3]
tmp_str = ["APPLE_1", "APPLE_2", "APPLE_3", "CAR_1", "CAR_2", "CAR_3", "DOG_1", "DOG_2", "DOG_3", "CHAIR_1", "CHAIR_2", "CHAIR_3"]
png = ".png"
for idx_j, j in enumerate(tmp):
    plt.imshow(j, cmap='hot', interpolation='nearest')
    plt.show()
    plt.savefig(tmp_str[idx_j]+png)

print("Hw | hW | hw  :  ", Hw, " | ", hW, " | ", hw)
