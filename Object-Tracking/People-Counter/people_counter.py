# importing the necessary packages
from utils.centroidtracker import CentroidTracker
from utils.trackableobject import TrackableObject 
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np 
import argparse
import imutils
import time
import dlib 
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required = True, help = "Path to Caffe deploy prototxt file")
ap.add_argument("-m", "--model", required = True, help = "Path to Caffe pre-trained model")
ap.add_argument('-i', "--input", type = str, help="path to optional video input file")
ap.add_argument("-o", "--output", type = str, help = "path to optional video output file")
ap.add_argument("-c", "--confidence", type = float, default = 0.4, help = "minimum probability to filter weak detections")
ap.add_argument("-s", "--skip-frames", type = int, default = 30, help = "# of skip frames between detections")
args = vars(ap.parse_args())

# initialize the list of class labels
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
            "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

# load our serialized model from the disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# if a video path was not supplied, grab a reference to the webcam
if not args.get("input", False):
    print("[INFO] Starting video stream...")
    vs = VideoStream(src = 0).start()
    time.sleep(2.0)

# otherwise, grab a reference to the video file
else:
    print("[INFO] opening the video file...")
    vs = cv2.VideoCapture(args["input"])


# initialize the video writer
writer = None

# initialize the frame dimensions
W = None
H = None

# instantiate our centroid tracker, then initialise a list to store 
# each of our dlib correlation trackers, followed by a dictionary to
# map each unique object ID to a TrackableObject
ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
trackers = [] 
trackableObjects = {}

# initialise the total number of frames processed, along with the
# total no.of objects that have either moved up/down
totalFrames = 0
totalDown = 0
totalUp = 0

# start the frames per second throughput estimator
fps = FPS().start()
