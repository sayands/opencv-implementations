# Importing necessary packages
import os
import sys
import cv2
import numpy as np 


# Creating the data matrix
def createDataMatrix(images):
    print('Creating data matrix', end = '...')

    numImages = len(images)
    sz = images[0].shape
    data = np.zeros((numImages, sz[0] * sz[1] * sz[2]), dtype = np.float32)
    for i in xrange(0, numImages):
        image = images[i].flatten()
        data[i,:] = image
    
    print("DONE")
    return data

# Creating new Face
def createNewFace(*args):
    output = averageFace

    # Add the eigenface with the weights
    for i in xrange(0, NUM_EIGEN_FACES):
        sliderValues[i] = cv2.getTrackbarPos("weight" + str(i), "Trackbars")
        weight = sliderValues[i] - MAX_SLIDER_VALUE /2
        output = np.add(output, eigenFaces[i] * weight)
    
    # Display Result at 2x size
    output = cv2.resize(output, (0,0), fx = 2, fy = 2)
    cv2.imshow("Result", output)

# Read images from the directory
def readImage():





if __name__ == '__main__':

    # Number of EigenFaces
    NUM_EIGEN_FACES = 10

    # Maximum weight
    MAX_SLIDER_VALUE = 255

    # Directory containing images
    dirName = "images"

    # Read images
    images = readImages(dirName)
    # Size of images
    sz = images[0].shape

    # Create data matrix for PCA
    data = createDataMatrix(images)

    # Compute the Eigenvectors from stack of images created
    print("Calculating PCA ", end = "...")
    mean, eigenvectors = cv2.PCACompute(data, mean = None, maxComponents = NUM_EIGEN_FACES)
    print("DONE")

    averageFace = mean.reshape(sz)

    eigenFaces = []

    for eigenVector in eigenvectors:
        eigenFace =eigenVector.reshape(sz)
        eigenFaces.append(eigenFace)

    # Create a window for displaying Mean Face
    cv2.namedWindow("Result", cv2.WINDOW_AUTOSIZE)

    # Display result at 2x size
    output = cv2.resize(averageFace, (0, 0), fx = 2, fy = 2)
    cv2.imshow(output)

    # Create Window for trackbars
    cv2.namedWindow("Trackbars", cv2.WINDOW_AUTOSIZE)

    sliderValues = []

    # Create Trackbars
    for i in xrange(0, NUM_EIGEN_FACES):
        sliderValues.append(MAX_SLIDER_VALUE/2)
        cv2.createTrackbar("weight" + str(i), "Trackbars", MAX_SLIDER_VALUE/2, MAX_SLIDER_VALUE, createNewFace)

    # Reset sliders by clicking on mean image
    cv2.setMouseCallback("Result", resetSliderValues)

    print('''Usage:
          Change the weight using the sliders. Click on
          result window to reset sliders...
          Hit ESC to terminate program''')
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

