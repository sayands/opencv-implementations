# import the necessary packages
from matplotlib import pyplot as plt 
import numpy as np 
import argparse
import cv2

# contruct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "Path to the image")
args = vars(ap.parse_args())

# load the image and show it
image = cv2.imread(args["image"])
cv2.imshow("image", image)

#----------------------------------------
# Creating a Grayscale image histogram
#----------------------------------------

# convert the image to grayscale and create a histogram
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("gray", gray)
hist = cv2.calcHist([gray], [0], None, [256], [0,256])
plt.figure()
plt.title('Grayscale Histogram')
plt.xlabel('Bins')
plt.ylabel("No.of pixels")
plt.plot(hist)
plt.xlim([0, 256])

#-----------------------------------------
# Creating a Flattened Color Histogram
#-----------------------------------------

# grab the image channels, initialise the tuple of colors,
# the figure and the flattened feature vector
chans = cv2.split(image) 
colors = ("b", "g", "r")
plt.figure()
plt.title("Flattened color histogram")
plt.xlabel("Bins")
plt.ylabel("No.of pixels")
features = []

# loop over the image channels
for (chan, color) in zip(chans, colors):
    # create a histogram for the current channel and
    # concentrate the resulting histograms for
    # each channel
    hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
    features.extend(hist)

    # plot the histogram
    plt.plot(hist, color = color)
    plt.xlim([0, 256])

# here we simply show the dimensionality of the 
# flattened color histogram 256 bins for each
# channel * 3 channels = 768 total values ---
# in practice we would normally not use 256 bins
# for each channel, a choice between 32-96 bins are 
# normally used, but this tends to be application
# dependent
print ("Flattened feature vector size: %d" % (np.array(features).flatten().shape))

# -------------------------------
# Computing a 2D color Histogram
#--------------------------------

# lets move on to 2D histogram -- reducing the 
# no.of bins from 256 to 32 so we can better
# visualise the results
fig = plt.figure()

# plot a 2D color histogram for green and blue
ax = fig.add_subplot(131)
hist = cv2.calcHist([chans[1],  chans[0]], [0,1], None, [32, 32], [0, 256, 0, 256])
p = ax.imshow(hist, interpolation = "nearest")
ax.set_title("2D color histogram for Green and Blue")
plt.colorbar(p)

# plot a 2D color histogram for green and red
ax = fig.add_subplot(132)
hist = cv2.calcHist([chans[1],  chans[2]], [0,1], None, [32, 32], [0, 256, 0, 256])
p = ax.imshow(hist, interpolation = "nearest")
ax.set_title("2D color histogram for Green and Red")
plt.colorbar(p)

# plot a 2D color histogram for blue and red
ax = fig.add_subplot(133)
hist = cv2.calcHist([chans[0],  chans[2]], [0,1], None, [32, 32], [0, 256, 0, 256])
p = ax.imshow(hist, interpolation = "nearest")
ax.set_title("2D color histogram for Blue and Red")
plt.colorbar(p)

# finally lets examine the dimensionality of one of the
# 2D histogram
print("2D histogram shape : %s with %d values" %(hist.shape, hist.flatten().shape[0]))
plt.show()


#------------------------------
# Creating a 3D color histogram
#------------------------------

hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
print("3D histogram shape: %s, with %d values" %(hist.shape, hist.flatten().shape[0]))