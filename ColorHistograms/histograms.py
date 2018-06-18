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
plt.show()