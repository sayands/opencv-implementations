# import the neccessary packages
from imutils.object_detection import non_max_suppression
import numpy as np 
import pytesseract
import argparse
import cv2

def decode_predictions(scores, geometry):
    # grab the no.of rows and columns from the score volume, then
    # initialise our set of bounding box rectangles and
    # corresponding confidence scores
    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []

    # loop over the no.of rows
    for y in range(0, numRows):
        # extract the scores(probability), followed by the
        # geometrical data used to derive potential bounding box
        # coordinates that surround text
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]

        # loop over the no.of columns
        for x in range(0, numCols):
            # if our score doesnt have sufficient probability, ignore it
            if scoresData[x] < args["min_confidence"]:
                continue
            
            # compute the offset factor as our resulting feature maps
            # will be 4x smaller than the input image
            (offsetX, offsetY) = ( x * 4.0, y * 4.0)

            # extract the rotation angle for the prediction and then
            # compute the sin and cosine
            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)

            # use the geometry volume to derive the width and height of the 
            # bounding box
            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]

            # compute both the starting and ending (x, y) - coordinates for
            # the text prediction bounding box
            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)

            # add the bounding box coordinates and probability score
            # to our respective lists
            rects.append((startX, startY, endX, endY))
            confidences.append(scoresData[x])

    # return a tuple of the bounding boxes and associated confidences
    return (rects, confidences)

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type = str, help = "Path to input image")
ap.add_argument("-east", "--east", type = str, help = "Path to input EASY text detector")
ap.add_argument("-c", "--min-confidence", type = float, default = 0.5, help = "minimum probability required to inspect a region")
ap.add_argument("-w", "--width", type = int, default = 320, help = "nearest multiple of 32 for resized width")
ap.add_argument("-e", "--height", type = int, default = 320, help = "nearest multiple of 32 for resized height")
ap.add_argument("-p", "--padding", type = float, default = 0.0, help = "amount of padding to add each border of ROI")
args = vars(ap.parse_args())

# load the input image and grab the image dimensions
image = cv2.imread(args["image"])
orig = image.copy()
(origH, origW) = image.shape[:2]

# set the new width and height and then determine the ratio in change
# for both the width and height
(newW, newH) = (args["width"], args["height"])
rW = origW / float(newW)
rH = origH / float(newH)

# resize the image and grab the new image dimensions
image = cv2.resize(image, (newW, newH))
(H, W) = image.shape[:2]

# define the output two layer names for the EAST detector model 
layerNames = [
    "feature_fusion/Conv_7/Sigmoid",
    "feature_fusion/concat_3"
]

# load the pre-trained EAST text detector
print("[INFO] loading EAST text detector...")
net = cv2.dnn.readNet(args["east"])

# construct the blob from the image and then forward pass of the
# model to obtain the two output layer sets
blob = cv2.dnn.blobFromImage(image, 1.0, (W, H), (123.68, 116.78, 103.94), swapRB = True, crop = False)
net.setInput(blob)
(scores, geometry) = net.forward(layerNames)

# decode the predictions, then apply NMS to suppress weak,
# overlapping bounding boxes
(rects, confidences) = decode_predictions(scores, geometry)
boxes = non_max_suppression(np.array(rects), probs = confidences)

# initialise the list of results
results = []