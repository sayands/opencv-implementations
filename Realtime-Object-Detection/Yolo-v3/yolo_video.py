# Import the neccessary packages
import numpy as np 
import argparse
import imutils
import time 
import cv2
import os 

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required = True, help = "Path to input video")
ap.add_argument("-o", "--output", required = True, help = "Path to output video")
ap.add_argument("-y", "--yolo", required = True, help = "base path to YOLO directory")
ap.add_argument("-c", "--confidence", type = float, default = 0.5, help = "Minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type = float, default = 0.3, help = "threshold when applying non-max suppression")
args = vars(ap.parse_args())

# load the COCO class labels our YOLO model was trained on
labelsPath = os.path.sep.join([args["yolo"], "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# initialise a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size = (len(LABELS), 3), dtype = "uint8")

# derive the paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join([args["yolo"], "yolov3.weights"])
configPath = os.path.sep.join([args["yolo"], "yolov3.cfg"])

# load our YOLO Object detector trained on COCO dataset
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# initialise the video stream, pointer to output video file and frame dimension
vs = cv2.VideoCapture(args["input"])
writer = None 
(W, H) = (None , None)

# try to determine the total no.of frames in the video file 
try:
    prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.iscv2() else cv2.CV_CAP_PROP_FRAME_COUNT
    total = int(vs.get(prop))
    print("[INFO] {} total frames in video".format(total))

except:
    print("[INFO] could not determin # of frames in video")
    print("[INFO] no approx. completion time can be provided")
    total -= 1
    
