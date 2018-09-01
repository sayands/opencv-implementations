# import the necessary packages
from imutils.video import VideoStream
from imutils import paths
import itertools
import argparse
import imutils
import time
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--models", required = True, help = "Path to directory containing neural style transfer models")
args = vars(ap.parse_args())

# grab the paths to all neural style transfer models
modelPaths = paths.list_files(args["models"], validExts = (".t7", ))
modelPaths = sorted(list(modelPaths))

# generate unique IDs for each of the model paths, then combine
# the two lists together
models = list(zip(range(0, len(modelPaths)), (modelPaths)))

modelIter = itertools.cycle(models)
(modelID, modelPath) = next(modelIter)

# loading the model from disk
print("[INFO] loading style transfer model...")
net = cv2.dnn.readNetFromTorch(modelPath)

# initialise the Video Stream
print("[INFO] starting video stream...")
vs = VideoStream(src = 0).start()
time.sleep(2.0)
print("[INFO] {}. {}".format(modelID + 1, modelPath))