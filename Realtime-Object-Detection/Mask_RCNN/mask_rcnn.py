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

# Draw the predicted bouding box, colorise and show the mask on image
def drawBox(frame, classId, conf, left, top, right, bottom, classMask):
    # Draw a bounding box
    cv.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 3)

    # Print a label of class
    label = '%.2f' % conf
    if classes:
        assert(classId < len(classes))
        label = '%s:%s' % (classes[classId], label)

    # Display the label at the top of the bounding box
    labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])
    cv.rectangle(frame, (left, top - round(1.5 * labelSize[1])), (left + round(1.5 * labelSize[0]), top + baseLine), (255, 255, 255), cv.FILLED)
    cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 1)

    # Resize the mask, threshold, color and apply on image
    classMask = cv.resize(classMask, (right - left + 1, bottom - top + 1))
    mask = (classMask > maskThreshold)
    roi = frame[top:bottom + 1, left:right + 1][mask]
    color = colors[classId % len(colors)]

    frame[top:bottom + 1, left:right + 1][mask] = ([0.5 * color[0], 0.5 * color[1], 0.5 * color[2]] + 0.5 * roi).astype(np.uint8)

# For each frame, extract the bounding box and mask for each detected object
def postprocess(boxes, masks):
    # Output size - N*C*H*W
    # N - no.of detected boxes
    # C - no.of classes(excluding background)
    # H*W - segmentation shape
    numClasses = masks.shape[1]
    numDetections = boxes.shape[2]

    frameH = frame.shape[0]
    frameW = frame.shape[1]

    for i in range(numDetections):
        box = boxes[0, 0, i]
        mask = masks[i]
        score = box[2]

        if score > confThreshold:
            classId = int(box[1])

            # Extract the bounding box 
            left = int(frameW * box[3])
            top = int(frameH * box[4])
            right = int(frameW * box[5])
            bottom = int(frameH * box[6])

            left = max(0, min(left, frameW - 1))
            top = max(0, min(top, frameH - 1))
            right = max(0, min(right, frameW - 1))
            bottom = max(0, min(bottom, frameH - 1))

            # Extract the mask for the object
            classMask = mask[classId]

            # Draw the bounding box, colorize and show the mask on image
            drawBox(frame, classId, score, left, top, right, bottom, classMask)



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
textGraph = "./model/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt";
modelWeights = "./model/frozen_inference_graph.pb";

# Loading the network
net = cv.dnn.readNetFromTensorflow(modelWeights, textGraph);
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