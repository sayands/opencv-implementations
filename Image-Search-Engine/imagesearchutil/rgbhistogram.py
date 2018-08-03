# import the necessary packages
import imutils
import cv2

class RGBHistogram:
    def __init__(self, bins):
        # store the no.of bins the histogram will use
        self.bins = bins

    def describe(self, image):
        # compute a 3D histogram in the RGB colorspace
        # then normalise the image
        hist = cv2.calcHist([image], [0, 1, 2], None, self.bins, [0, 256, 0, 256, 0, 256])
            
        hist = cv2.normalize(hist, hist)

        return hist.flatten()