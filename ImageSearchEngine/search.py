# import the necessary packages
from imagesearchutil.searcher import Searcher
import numpy as np 
import argparse
import os
import pickle
import cv2 

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required = True, help = "Path to the directory that contains the images we just indexed")
ap.add_argument("-i", "--index", required = True, help = "Path to where we stored our index")
args = vars(ap.parse_args())

# load the index and initialise our searcher
index = pickle.loads(open(args["index"], "rb").read())
searcher = Searcher(index)

# loop over the images in the index -- we 
# will use each one as a query image
for (query, queryFeatures) in index.items():
    # perform the search using current query
    results = searcher.search(queryFeatures)

    # load the query image and display it
    path = os.path.join(args["dataset"], query)
    queryImage = cv2.imread(path)
    cv2.imshow("Query", queryImage)
    print("query: {}".format(query))

    # initialise the two montages to display our results --
    # we have a total of 25 images in the index but lets only
    # display the top 10 results; 5 images per montage, with
    # images that are 400 * 166 pixels

    montageA = np.zeros(( 166 * 5, 400 ,3), dtype = "uint8")
    montageB = np.zeros(( 166 * 5, 400, 3), dtype = "uint8")

    # loop over the top ten results
    for j in range(0, 10):
        # grab the result( we are using row-major order) and
        # load the result image
        ( score, imageName ) = results[j]
        path = os.path.join(args["dataset"], imageName)
        result = cv2.imread(path)
        print("\t{}. {} : {:.3f}".format(j+1, imageName, score))

        # check to see if the first montage should be used
        if j < 5 :
            montageA[ j * 166 : (j+1)*166, : ] = result
        
        # otherwise second montage should be used
        else:
            montageB[(j - 5) * 166 : ((j-5) + 1) * 166, :] = result
    
    # show the results
    cv2.imshow("Results 1-5", montageA)
    cv2.imshow("Results 6-10", montageB)
    cv2.waitKey(0)