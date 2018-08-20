# import the necessary packages
from imutils.object_detection import non_max_suppression
import numpy as np 
import argparse
import time
import cv2

# construct the argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str, help = "path to input image")
ap.add_argument("-east", "--east", type = str, help = "path to input EAST text detection")
ap.add_argument("-c", "--min-confidence", type=float, default = 0.5, help = "minimum probability required to inspect a region")
ap.add_argument("-w", "--width", type=int, default = 320, help = "resized image width(should be mulitple of 32)")
ap.add_argument("-h", "--height", type=int, default = 320, help = "resized image height(should be mulitple of 32)")
args = vars(ap.parse_args())

# load the input image and grab the image dimensions
image = cv2.imread(args["image"])
orig = image.copy()
(H, W) = image.shape[:2]

# set the new width and height and then determine the ratio in change
# for both width and height
(newW, newH) = (args["width"], args["height"])
rW = W / float(newW)
rH = H / float(newH)

# resize the image and grab the new image dimensions
image = cv2.resize(image, (newW, newH))
(H, W) = image.shape[:2]

# define the two output layer names for the EAST detector model
# first is the output probabilities and the second can be used to 
# derive bounding box coordinates of text
layerNames = [
    "feature_fusion/Conv_7/Sigmoid",
    "feature_fusion/concat_3"]
