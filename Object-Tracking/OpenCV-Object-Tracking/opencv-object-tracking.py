# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
import argparse
import imutils
import time
import cv2

# construct the argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", type = str, help = "path to input video file")
ap.add_argument("-t", "--tracker", type = str, default = "kcf", help = "OpenCV Object Tracking type")
args = vars(ap.parse_args())

# extract the OpenCV version info
(major, minor) = cv2.__version__.split(".")[:2]

# create object tracker in accordance with the version of OpenCV
if int(major) == 3 and int(minor)<3:
    tracker = cv2.Tracker_create(args["tracker"].upper())

else:
    OPENCV_OBJECT_TRACKERS = {
        "csrt" : cv2.TrackerCSRT_create,
        "kcf"  : cv2.TrackerKCF_create,
        "boosting" : cv2.TrackerBoosting_create,
        "mil" : cv2.TrackerMIL_create,
        "tld" : cv2.TrackerTLD_create,
        "medianflow" : cv2.TrackerMedianFlow_create,
        "mosse" : cv2.TrackerMOSSE_create
    }

    tracker =  OPENCV_OBJECT_TRACKERS[args["tracker"]]()

# initialize the bounding box coordinates of the object we are going to track
initBB = None

# if a video path was not supplied, grab reference to the webcam
if not args.get("video", False):
    print("[INFO] starting video stream...")
    vs = VideoStream(src = 0).start()
    time.sleep(1.0)

else:
    vs = cv2.VideoCapture(args["video"])

# initialise the FPS throughput estimator
fps = None

# loop over the frames in the video stream
while True:
    # grab the current frame and handle it
    frame = vs.read()
    frame = frame[1] if args.get("video", False) else frame

    # check to see if we have reached end of the stream
    if frame is None:
        break
    
    # resize the frame for faster processing and grab the frame dimensions
    frame = imutils.resize(frame, width = 500)
    (H, w) = frame.shape[:2]

    # check to see if we are currently tracking an object
    if initBB is not None:
        # grab the new bounding box coordinates of the object
        (success, box) = tracker.update(frame)

        # check to see if the tracking success was a success
        if success:
            (x, y, w, h) = [int(v) for v in box]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # update the FPS counter
        fps.update()
        fps.stop()

        # initialise the set of information we will be displaying on the frame
        info = [
            ( "Tracker", args["tracker"]),
            ("Success", "Yes" if success else "No"),
            ("FPS", "{:.2f}".format(fps.fps()))
        ]

        # loop over the info tuples and display on frame
        for (i, (k, v)) in enumerate(info):
            text = "{}: {}".format(k, v)
            cv2.putText(frame, text, (10, H - ((i * 20) + 20)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    # show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the 's' key is selected, we are going to "select" a bounding
    # box to track
    if key == ord("s"):
        # select the bounding box of the object we want to track
        initBB = cv2.selectROI("Frame", frame, fromCenter = False)

        # start OpenCV Object Tracker using the supplied bounding box 
        # coordinates, then start the FPS throughput estimator as well
        tracker.init(frame, initBB)
        fps = FPS().start()
    
    elif key == ord("q"):
        break

# release webcam pointer
if not args.get("video", False):
    vs.stop()

# else, release the file pointer
else:
    vs.release()

# close all windows
cv2.destroyAllWindows()


    