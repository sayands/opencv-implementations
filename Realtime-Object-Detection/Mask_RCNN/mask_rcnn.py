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

# Reading the input
outputFile = 'mask_rcnn_out_py.avi'
if(args.image):
    # Open the image file
    if not os.path.isfile(args.image):
        print('Input Image File', args.image, " doesnt exist")
        sys.exit(1)
    
    cap = cv.VideoCapture(args.image)
    outputFile = args.image[:-4] + '_mask_rcnn_out_py.jpg'

elif(args.video):
    # Open the video file
    if not os.path.isfile(args.video):
        print("Input video file", args.video, " doesnt exist")
        sys.exit(1)

    cap = cv.VideoCapture(args.video)
    outputFile = args.video[:-4] + '_mask_rcnn_out_py.avi'

else:
    # Webcam Input
    cap = cv.VideoCapture(0)

# Get the video writer initialised to save the output video
if(not args.image):
    vid_writer = cv.VideoWriter(outputFile, cv.VideoWriter_fourcc('M', 'J', 'P', 'G'), 28, (round(cap.get(cv.CAP_PROP_FRAME_WIDTH)), round(cap.get(cv.CAP_PROP_FRAME_HEIGHT))))

# Processing each frame
while cv.waitKey(1) < 0:
    # Get frame from the video
    hasFrame, frame = cap.read()

    # Stop the program if reached end of video
    if not hasFrame:
        print('Done processing')
        print('Output File is stored as ', outputFile)
        cv.waitKey(3000)
        break

    # Create a 4D blob from a frame
    blob = cv.dnn.blobFromImage(frame, swapRB = True, crop = False)

    # Set the input to the network
    net.setInput(blob)

    # Run the forward pass to get output from the output layers
    boxes, masks = net.forward(['detection_out_final', 'detection_masks'])

    # Extract the bounding box and mask for each of the detected objects
    postprocess(boxes, masks)

    # Put efficiency information
    t, _ = net.getPerfProfile()
    label = 'Mask-RCNN : Inference time: %.2f ms' % ( t * 1000.0 / cv.getTickFrequency())
    cv.putText(frame, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))

    # Write the frame with detection boxes
    if(args.image):
        cv.imwrite(outputFile, frame.astype(np.unint8))
    else:
        vid_writer.write(frame.astype(np.unint8))
    
    cv.imshow(winName, frame)