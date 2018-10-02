# Importing necessary packages and libraries
import cv2 as cv
import argparse
import numpy as np 
import os.path 
import sys 
import random 

# Initialising the parameters
confThreshold = 0.5
maskThreshold = 0.3

# Constructing the argument Parser
parser = argparse.ArgumentParser()
parser.add_argument('--image', help = 'Path to image file')
parser.add_argument('--video', help = 'Path to video file')
args = parser.parse_args()

# Loading names of classes
classesFile = 'model/mscoco_labels.names'
classes = None 
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

# Loading the colors
colorsFile = 'model/colors.txt'
with open(colorsFile, 'rt') as f:
    colorsStr = f.read().rstrip('\n').split('\n')
colors = []
for i in range(len(colorsStr)):
    rgb = colorsStr[i].split(' ')
    color = np.array([float(rgb[0]), float(rgb[1]), float(rgb[2])])
    colors.append(color)

# Loading the textGraph nad weight files for the model
textGraph = "./model/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt"
modelWeights = "./model/frozen_inference_graph.pb"

# Loading the network
net = cv2.dnnreadNetFromTensorflow(modelWeights, textGraph)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
