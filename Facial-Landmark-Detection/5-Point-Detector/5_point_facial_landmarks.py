# Importing necessary packages
from imutils.video import VideoStream
from imutils import face_utils
import argparse
import imutils
import time
import dlib 
import cv2

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument('-p', '--shape-prediction', required = True, help = "Path to facial landmark predictor")
args = vars(ap.parse_args())

# Initialise dlib's face detector(HOG-based) and then create the facial
# landmark predictor
print("[INFO] Loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predcitor(args['shape-predictor'])

# Initialise the Video Stream and sleep for a bit, allowing the camera sensor 
# to warm up
print("[INFO] Camera Sensor Warming Up...")
vs = VideoStream(src=1).start()
time.sleep(2.0)

# loop over the frames from the video stream
while True:
    # grab the frame from the threaded video stream, resize it to 
    # have a maximum width of 400 pixels, and convert it to grayscale
    frame = vs.read()
    frame = imutils.resize(frame, width = 400)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect faces in the grayscale frame
    rects = detector(gray, 0)

    # check to see if a face was detected, and if so, draw the total 
    # number of faces on the frame time
    if len(rects) > 0:
        text = "{} face(s) found".format(len(rects))
        cv2.putText(frame, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)