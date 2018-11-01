# Importing the neccessary packages
from imutils.video import FPS
import multiprocessing
import numpy as np 
import argparse
import imutils
import dlib 
import cv2 

# Function for spawning new process
def start_tracker(box, label, rgb, inputQueue, outputQueue):
    # construct a dlib rectangle object from the bounding box
    # coordinates and then start the correlation tracker
    t = dlib.correlation_tracker()
    rect = dib.rectangle(box[0], box[1], box[2], box[3])
    t.start_track(rgb, rect)

    # indefinite loop
    while True:
        rgb = inputQueue.get()

        if rgb is not None:
            # update the tracker and grab the position of the tracked object
            t.update(rgb)
            pos = t.get_position()

            # unpack the position object
            startX = int(pos.left())
            startY = int(pos.top())
            endX = int(pos.right())
            endY = int(pos.bottom())

            # add the label - bounding box coordinates to the output queue
            outputQueue.put((label, (startX, startY, endX, endY)))

# construct the argument parser and parse the arguments
parser = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required = True, help = "path to Caffe deploy prototxt file")
ap.add_argument("-m", "--model", required = True, help = "path to Caffe pre-trained model")
ap.add_argument("-v", '--video', required = True, help = "path to input video file")
ap.add_argument("-o", "--output", type = str, help = "path to optional video file")
ap.add_argument("-c", "--confidence", type = float, default = 0.2, help = "minimum probability to filter out weak detections")
args = vars(ap.parse_args())

# initialize the list of queues - both input queue and output queue
inputQueue = []
outputQueues = []

# initialise the list of class labels MobileNet SSD was trained to detect
CLASSES = ['background', 'aeroplane','bicycle', 'bird', 'bottle', 'bus', 'car', 'cat', 'chair',
        'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
        'train', 'tvmonitor']

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args['model'])

# initialise the video stream and output video writer
print("[INFO] starting video stream...")
vs = cv2.VideoCapture(args['video'])
writer = None

# start the frames per second throughput estimator
fps = FPS().start()

# loop over frames from the video file stream
while True:
    (grabbed, frame) = vs.read()

    if frame is None:
        break
    
    frame = imutils.resize(frame, width = 600)
    rgb = cv2.cvtColor(frame, COLOR_BGR2RGB)

    if args['output'] is not None and writer is None:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(args["output"], fourcc, 30, (frame.shape[1], frame.shape[0]), True)
    
    if len(inputQueues) == 0:
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 0.007843, (w.h), 127.5)

        net.setInput(blob)
        detections = net.forward()

        for i in np.arange(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > args['confidence']:
                idx = int(detections[0, 0, i, 1])
                label = CLASSES[idx]

                if CLASSES[idx]!= 'person':
                    continue