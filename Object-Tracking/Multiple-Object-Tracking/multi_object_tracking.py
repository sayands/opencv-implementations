# importing the necessary packages
from imutils.video import VideoStream
import argparse
import time
import cv2
import imutils

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", type = str, help ="Path to input video file")
ap.add_argument("-t", "--tracker", type = str, default = "kcf", help = "OpenCV object tracker type")
args = vars(ap.parse_args())

# initialise a dictionary that maps strings to their corresponding
# OpenCV object tracker implementations
OPENCV_OBJECT_TRACKERS = {
    'csrt' : cv2.TrackerCSRT_create,
    'kcf'  : cv2.TrackerKCF_create,
    'boosting' : cv2.TrackerBoosting_create,
    'mil' : cv2.TrackerMIL_create,
    'tld' : cv2.TrackerTLD_create,
    'medianflow' : cv2.TrackerMedianFlow_create,
    'mosse' : cv2.TrackerMOSSE_create
}

# initialise OpenCV's special multi-object tracker
trackers = cv2.MultiTracker_create()

# if a video path was not supplied, grab the reference to a web cam
if not args.get("video", False):
    print("[INFO] Starting video stream...")
    vs = VideoStream(src = 0).start()
    time.sleep(1.0)

# otherwise grab a reference to the video file
else:
    vs = cv2.VideoCapture(args["video"])