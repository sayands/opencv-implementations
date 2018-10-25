# Importing the necessary packages
from imutils.video import FPS
import numpy as np 
import argparse
import imutils
import dlib
import cv2 

# Construct the argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required = True, help = 'Path to Caffe deploy prototxt file')
ap.add_argument("-m", "--model", required = True, help = "Path to Caffe pre-trained model")
ap.add_argument("-v", "--video", required = True, help = "Path to input video file")
ap.add_argument("-l", "--label", required = True, help = "Class label we are interested in detecting + tracking")
ap.add_argument("-o", "--output", type = str, help ="Path to optional output video file")
ap.add_argument("-c", "--confidence", type = float, default = 0.2, help = "minimum probability to filter weak detections")
args = vars(ap.parse_args())

# Initialise the list of class labels MobileNet SSD was trained to detect
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", 
            "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# Some more initialisations
print("[INFO] starting video stream...")
vs = cv2.VideoCapture(args["video"])
tracker = None 
writer = None 
label = ""

# start the FPS throughput estimator
fps = FPS().start()

# loop over frames in the Video File Stream
while True:
    (grabbed, frame) = vs.read()

    if frame is None:
        break
    
    frame = imutils.resize(frame, width = 600)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # if writing output video to disk
    if args['output'] is not None and writer is None:
        fourcc = cv2.VideoWriter_fourcc("MJPG")
        writer = cv2.VideoWriter(args['output'], fourcc, 30, (frame.shape[1], frame.shape[0]), True)
    