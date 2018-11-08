# Import the neccessary packages
from imutils import face_utils
from imutils import paths 
import numpy as np 
import argparse 
import imutils 
import shutil 
import json 
import dlib 
import cv2 
import os 
import sys

def overlay_image(bg, fg, fgMask, coords):
    # grab the foreground spatial dimensions
    (sH, sW) = fg.shape[:2]
    (x, y) = coords

    overlay = np.zeros(bg.shape, dtype = "uint8")
    overlay[y:y + sH, x: x + sW] = fg 

    alpha = np.zeros(bg.shape[:2], dtype = "uint8")
    alpha[y:y + sH, x: x + sW] = fgMask
    alpha = np.dstack([alpha] * 3)

    # perform alpha blending to merge background, foreground and alpha channel
    output = alpha_blend(overlay, bg, alpha)

    return output 

def alpha_blend(fg, bg, alpha):
    fg = fg.astype("float")
    bg = bg.astype("float")
    alpha = alpha.astype("float") / 255 

    # perform alpha blending
    fg = cv2.multiply(alpha, fg)
    bg = cv2.multiply(1 - alpha, bg)

    # add the foreground and background to obtain the final output image
    output = cv2.add(fg, bg)

    # return the output image
    return output.astype("uint8")

def create_gif(inputPath, outputPath, delay, finalDelay, loop):
    # grab all image paths in the input directory
    imagePaths = sorted(list(paths.list_images(inputPath)))

    # remove the last image in the list
    lastPath = imagePaths[-1]
    imagePaths = imagePaths[:-1]

    cmd = "convert -delay {} {} -delay {} {} -loop {} {}".format(delay, " ".join(imagePaths), finalDelay, lastPath, loop, outputPath)
    os.system(cmd)

# Construct the argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--config", required = True, help = "Path to configuration file")
ap.add_argument("-i", "--image", required = True, help = "Path to input image")
ap.add_argument("-o", "-output", required = True, help = "Path to output GIF")
args = vars(ap.parse_args())

# load the JSON configuration file
config = json.loads(open(args["config"]).read())
sg = cv2.imread(config["sunglasses"])
sgMask = cv2.imread(config["sunglasses_mask"])

shutil.rmtree(config["temp_dir"], ignore_errors = True)
os.makedirs(config["temp_dir"])

# load OpenCV face detector and dlib facial landmark predictor
print("[INFO] loading models...")
detector = cv2.dnn.readNetFromCaffe(config["face_detector_prototxt"], config["face_detector_weights"])
predictor = dlib.shape_predictor(config["landmark_predictor"])

# load the input image and construct an input blob from the image 
image = cv2.imread(args["image"])
(H, W) = image.shape[:2]
blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

# pass the blob through the network and obtain the detections
print("[INFO] computing object detection...")
detector.setInput(blob)
detections = detector.forward()

i = np.argmax(detections[0, 0, :, 2])
confidence = detections[0, 0, i, 2]

# filter out weak detections
if confidence < config["min_confidence"]:
    print("[INFO] no reliable faces found")
    sys.exit(0)

# compute the (x, y)-coordinates of the bounding box for the face
box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
(startX, startY, endX, endY) = box.astype("int")

# construct a dlib rectangle object from bounding box coordinates and 
# determine the facial coordinates
rect = dlib.rectangle(int(startX), int(startY), int(endX), int(endY))
shape = predictor(image, rect)
shape = face_utils.shape_to_np(shape)

# grab the indexes of the facial landmarks for the left and right eye 
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
leftEyePts = shape[lStart:lEnd]
rightEyePts = shape[rStart:rEnd]

# compute the center of mass for each eye