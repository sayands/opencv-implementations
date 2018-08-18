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

# loop over frames from the video stream
while True:
    # grab the next frame and handle if we are reading from either 
    # VideoCapture or VideoStream
    frame = vs.read()
    frame = frame[1] if args.get("input", False) else frame

    if args["input"] is not None and frame is None:
        break
    
    # resize the frame and convert from BGR to RGB
    frame = imutils.resize(frame, width = 500)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    if W is None or H is None:
        (H, w) = frame.shape[:2]
    
    # if we are writing to an output file on the disk
    if args["output"] is not None and writer is None:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(args["output"], fourcc, 30, (W, H), True)
    
    # initialise the current status alongwith our list of bounding
    # box rectangles returned by object detector or correlation
    # trackers
    status = "Waiting"
    rects = []

    # check to see if we should run object detector to aid tracker
    if totalFrames % args["skip_frames"] == 0:
        status = "Detecting"
        trackers = []

        # convert the frame to a blob and pass the blob through the
        # network and obtain the detections
        blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
        net.setInput(blob)
        detections = net.forward()


        # loop over the detections
        for i in np.arange(0, detections.shape[2]):
            # extract the confidence associated with each prediction
            confidence = detections[0, 0, i, 2]

            # filter out weak detections by requiring a minimum confidence
            if confidence > args["confidence"]:
                # extract the index of the class label from the detections list
                idx = int(detections[0, 0, i, 1])

                # if the class label is not a person, ignore it
                if CLASSES[idx] != "person":
                    continue
                
                # compute the (x, y) coordinates of the bounding box
                # for the object
                box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                (startX, startY, endX, endY) = box.astype("int")

                # construct a dlib rectangle object from the bounding box
                # coordinates and then start the dlib correlation tracker
                tracker = dlib.correlation_tracker()
                rect = dlib.rectangle(startX, startY, endX, endY)
                tracker.start_track(rgb, rect)

                # add the tracker to our list of trackers so we can
                # utilise it during skip frames
                trackers.append(tracker)
