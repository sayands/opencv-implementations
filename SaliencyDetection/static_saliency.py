# import the neccessary packages
import argparse
import cv2
import numpy as np

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "Path to input image")
args = vars(ap.parse_args())

# load the image 
image = cv2.imread(args["image"])

# initialise OpenCV's static saliency spectral residual detector and
# compute the saliency map
saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
(success, saliencyMap) = saliency.computeSaliency(image)
saliencyMap = (saliencyMap * 255).astype("uint8")
cv2.imshow("Image", image)
cv2.imshow("Output", saliencyMap)
cv2.waitKey(0)

# initialise OpenCV's static fine grained saliency detector and
# compute the saliency map
saliency = cv2.saliency.StaticSaliencyFineGrained_create()
(success, saliencyMap) = saliency.computeSaliency(image)

# threshhold the saliencyMap
threshMap = cv2.threshold(saliencyMap, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

# perform a series of erosions and dilations using an elliptical kernel
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
skinMask = cv2.erode(threshMap, kernel, iterations = 2)
skinMask = cv2.dilate(skinMask, kernel, iterations = 2)

# blur mask to remove noise and then applly blur to image
skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)
skin = cv2.bitwise_and(image, image, mask = skinMask)

# show the image, saliencyMap and threshMap
cv2.imshow("Image", image)
cv2.imshow("Output", saliencyMap)
cv2.imshow("Thresh", threshMap)
cv2.waitKey(0)

#show the image, saliencyMap and skin in image
cv2.imshow("Image", image)
cv2.imshow("Output", saliencyMap)
# show the skin in the image along with the mask
cv2.imshow("Image", np.hstack([image, skin]))
cv2.waitKey(0)