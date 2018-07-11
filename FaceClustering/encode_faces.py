# import the necessary packages
from imutils import paths
import face_recognition
import argparse
import pickle
import cv2
import os

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--dataset", required = True, help = "Path to the input directory of faces + images")
ap.add_argument("-e", "--encodings", required = True, help = "Path to serialized db of facial encodings")
ap.add_argument("-d", "--detection-method", type = str, default = "hog", help = "face detection model to use : either hog or cnn")
args = vars(ap.parse_args())