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

# loop over frames from the video File Stream
while True:
    frame = vs.read()

    frame = imutils.resize(frame, width = 600)
    orig = frame.copy()
    (h, w) = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(frame, 1.0, (w, h), (103.969, 116.779, 123.860), swapRB = False, crop = False)
    net.setInput(blob)
    output = net.forward()

    # reshape the output tensor, add the mean subtraction and then swap the channel ordering
    output = output.reshape((3, output.shape[2], output.shape[3]))
    output[0] += 103.969
    output[1] += 116.779
    output[2] += 123.680
    output /= 255.0
    output = output.transpose(1, 2, 0)

    # show the original frame alongwith output neural style
    cv2.imshow("Input", frame)
    cv2.imshow("Output", output)
    key = cv2.waitKey(1) & 0xFF

    # if the 'n' key is pressed, load the next neural style transfer model
    if key == ord('n'):
        # grab the next neural transfer model and load it
        (modelID, modelPath) = next(modelIter)
        print('[INFO] {}. {}'.format(modelID + 1, modelPath))
        net = cv2.dnn.readNetFromTorch(modelPath)

    elif key == ord('q'):
        break

# cleanup
cv2.destroyAllWindows()
vs.stop()

