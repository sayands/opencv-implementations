# import the necessary packages
from imagesearchutil.rgbhistogram import RGBHistogram 
from imutils.paths import list_images
import argparse
import pickle
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required= True, help = "path to the directory")
ap.add_argument("-i", "--index", required = True, help = "Path to where the computed index will be stored")
args = vars(ap.parse_args())

# initialise the index dictionary to store our quantified
# images, with the 'key' of the dictionary being the image
# filename and the 'value' of our computed features
index = {}

# initialising our image descriptor with 8 bins per channel
desc = RGBHistogram([8, 8, 8])

# use list_images to grab the image paths and loop over
# them
for imagePath in list_images(args["dataset"]):
    # extract our unique image ID,i.e, the filename
    k = imagePath[imagePath.rfind("/") + 1 :]

    # load the image, describe it using our RGB histogram
    # descriptor, and update the index
    image = cv2.imread(imagePath)
    features = desc.describe(image)
    index[k] = features

# we are now done indexing our image - now we can write our
# index to disk
f = open(args["index"], "wb")
f.write(pickle.dumps(index))
f.close()

# show how many images we indexed
print("[INFO] done...indexed {} images".format(len(index)))
