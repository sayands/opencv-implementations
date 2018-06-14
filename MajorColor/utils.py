# import the necessary packages
import numpy as np 
import cv2

def centroid_histogram(clt):
    # grab the number of different clusters and create a histogram
    # based on the no.of pixels assigned to each cluster
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _ ) = np.histogram(clt.labels_, bins = numLabels)

    # normalize the histogram, such that it sums to one
    hist = hist.astype('float')
    hist /= hist.sum()

    # return the histogram
    return hist

def plot_color(hist, centroids):
    # initialise the bar chart representing the
    # relative frequency of each of the colors
    bar = np.zeros((50, 300, 3), dtype="uint8")
    startX = 0

    # loop over the percentage of each cluster and color
    # of each cluster
    for (percent, color) in zip(hist, centroids):
        # plot the relative percentage of each cluster
        endX = startX + ( percent * 300)
        cv2.rectangle(bar, (int(startX),0), (int(endX), 50), color.astype("uint8").tolist(), -1)
        startX = endX
    
    # return the bar chart
    return bar

# build a histogram of clusters and then create a figure 
# representing the number of pixels labeled to each color