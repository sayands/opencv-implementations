#import the neccessary packages
import numpy as np 
import argparse
import cv2 

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument('-i', "--image", required = True, help = "Path to input image file")
args = vars(ap.parse_args())

# load the image from disk
image = cv2.imread(args["image"])

# convert the image to grayscale and flip foreground and
# background 
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.bitwise_not(gray)

# threshold the image, setting all foreground pixels to
# 255 and all background pixels to 0
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

# grab the (x,y) coordinates of all pixel values that are greater than zero, then
# use these coordinates to compute a rotated bounding box that contains all coordinates
coords = np.column_stack(np.where(thresh > 0))
angle = cv2.minAreaRect(coords)[-1]

if angle < -45:
    angle = -(90 + angle)

else :
    angle = -angle

# rotate the image to deskew it
(h, w) = image.shape[:2]
center = (w // 2, h // 2)
M = cv2.getRotationMatrix2D(center, angle, 1.0)
rotated = cv2.warpAffine(image, M, (w,h), flags = cv2.INTER_CUBIC, borderMode = cv2.BORDER_REPLICATE)

# draw the correction angle on the image so we can validate it
cv2.putText(rotated, "Angle: {:.2f} degrees".format(angle), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

# show the output image
print("[INFO] angle: {:.3f}".format(angle))
cv2.imshow("Input", image)
cv2.imshow("Rotated", rotated)
cv2.waitKey(0)