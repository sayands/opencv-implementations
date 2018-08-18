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
    # otherwise, we should utilize our object tracker rather than
    # object detector to obtain a higher frame processing throughput
    else:
        # loop over the trackers
        for tracker in trackers:
            # set the status of our system to be 'tracking' rather
            # than 'waiting' or 'detecting'
            status = "Traacking"

            # update the tracker and grab the updated position
            tracker.update(rgb)
            pos = tracker.get_position()

            # unpack the position object
            startX = int(pos.left())
            startY = int(pos.top())
            endX = int(pos.right())
            endY = int(pos.bottom())

            # add the bounding box coordinates to the rectangles list
            rects.append((startX, startY, endX, endY))
    
    # draw a horizontal line in the center of the frame -- once an
    # object crosses this line we will determine whether they were 
    # moving 'UP' or 'DOWN'
    cv2.line(frame, (0, H // 2), (W, H // 2), (0, 255, 255), 2)

    # use the centroid tracker to associate the old object
    # centroids with newly computed object centroids
    objects = ct.update(rects)

    # loop over the tracked objects
    for (objectID, centroid) in objects.items():
        # check to see if a trackable object exists for the current
        # object ID
        to = trackableObjects.get(objectID, None)

        # otherwise, there is a trackable object so we can utilize 
        # it to determine direction
        if to is None:
            to = TrackableObject(objectID, centroid)
        
        else:
            # difference between y-coordinates of the *current* centroid
            # and the mean of *previous* centroids will tell us in which
            # direction the object is moving ( negative for 'up' and 
            # position for 'down')
            y = [c[1] for c in to.centroids]
            direction = centroid[1] - np.mean(y)
            to.centroids.append(centroid) 

            # check to see if the object has been counted or not
            if not to.counted:
                # if the direction is negative(indicating the object is
                # moving up) AND the centroid is above the center line
                # Count the object
                if direction < 0 and centroid[1] < H // 2:
                    totalUp += 1
                    to.counted = True

                elif direction > 0 and centroid[1] > H // 2:
                    totalDown += 1
                    to.counted = True

        # store the trackable object in our dictionary
        trackableObjects[objectID] = to

        # draw both the ID of the object and the centroid of the
        # object on the output frame
        text = "ID {}".format(objectID)
        cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
    

    # construct a tuple of information we will be displaying on the frame
    info = [
        ("Up", totalUp),
        ("Down", totalDown),
        ("Status", status),
    ]

    # loop over the info tuples and draw them on our frame
    for (i, (k, v)) in enumerate(info):
        text = "{}: {}".format(k, v)
        cv2.putText(frame, text, (10, H - ((i * 20) + 20)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # check to see if we should write frame to disk
    if writer is not None:
        writer.write(frame)
    
    # show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the 'q' key was pressed, break from the  loop
    if key == ord('q'):
        break
    
    # increment the total no.of frames processed thus far and
    # then update the FPS counter
    totalFrames += 1
    fps.update()

# stop the counter and display FPS information
fps.stop()
print("[INFO] elapsed time: {: .2f}".format(fps.elapsed()))
print("[INFO] approx FPS: {: .2f}".format(fps.fps()))

if writer is not None:
    writer.release()

if not args.get("input", False):
    vs.stop()
else
    vs.release()

# close any open windows
cv2.destroyAllWindows()