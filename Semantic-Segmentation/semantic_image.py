# import the necessary packages
import numpy as np 
import argparse
import imutils
import time
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required = True, help = "Path to deep learning segmentation model")
ap.add_argument("-c", "--classes", required = True, help = "Path to .txt file containing class labels")
ap.add_argument("-i", "--image", required= True, help = "Path to input image")
ap.add_argument("-l", "--colors", type= str, help = "Path to .txt file containing colors for labels")
ap.add_argument("-w", "--width", type=int, default = 500, help = "desired width(in pixels) of input image")
args = vars(ap.parse_args())

# load the class label names
CLASSES = open(args["classes"]).read().strip().split("\n")

# if a colors file was supplied, load it from disk
if args['colors']:
    COLORS = open(args["colors"]).read().strip().split("\n")
    COLORS = [np.array(c.split(",")).astype("int") for c in COLORS]
    COLORS = np.array(COLORS, dtype="uint8")

# otherwise, we need to randomly generate RGB colors for each class label
else:
    # initialize a list of colors to represent each class label in
    # the mask( starting with 'black' for the background/unlabeled
    # regions)
    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size = (len(CLASSES) -1, 3), dtype="uint8")
    COLORS = np.vstack([[0, 0, 0], COLORS]).astype("uint8")

#  initialize the legend visualisation
legend = np.zeros(((len(CLASSES) * 25) + 25, 300, 3), dtype="uint8")

# loop over the class names + colors
for (i, (className, color)) in enumerate(zip(CLASSES, COLORS)):
    # draw the class name + color on the legend
    color = [int(c) for c in color]
    cv2.putText(legend, className, (5, (i * 25) + 17), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.rectangle(legend, (100, (i * 25)), (300, (i * 25) + 25), tuple(color), -1)
    