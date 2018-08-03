# import the necessary packages
import cv2

#load the image and show it
image = cv2.imread('jurassic-park-tour-jeep.jpg')
cv2.imshow('Original', image)
cv2.waitKey(0)

print (image.shape)

# Making the image 100 pixels wide
r = 100.0/image.shape[1]
dim = (100, int(image.shape[0] * r))

# perform resizing
resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
cv2.imshow("resized", resized)
cv2.waitKey(0)

# grab dimensions of the image
(h, w) = image.shape[:2]
center = (w/2, h/2)

# rotate the image by 180 degrees
M = cv2.getRotationMatrix2D(center, 180, 1.0)
rotated = cv2.warpAffine(image, M, (w,h))
cv2.imshow("rotated", rotated)
cv2.waitKey(0)

# crop the image
cropped = image[70:170, 440:540]
cv2.imshow("cropped", cropped)
cv2.waitKey(0)

# writing image to disk
cv2.imwrite("thumbnail.png", cropped)

