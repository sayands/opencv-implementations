from scipy.spacial import distance as distance
from collections import OrderedDict
import numpy as np 

class CentroidTracker():
    def __init__(self, maxDisappeared = 50):
        # initialise the next unique ObjectID alongwith two
        # ordered dictionaries to keep track of mapping a given object
        # ID to its centroid and no.of consecutive disappeared frames
        self.nextObjectID = 0;
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()

        self.maxDisappeared = maxDisappeared
    
    def register(self, centroid):
        # register the next available object ID to store the centroid
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID +=1

    def dereigister(self, objectID):
        # to deregister an object ID we delete the object ID from
        # both of our respective dictionaries
        del self.objects[objectID]
        del self.disappeared[objectID]
    
    def update(self, rects):
        # check to see if list of input bounding vox rectangles are
        # empty
        if len(rects) == 0:
            # loop over any existing tracked objects and mark them as
            # disappeared
            for objectID in self.disappeared.keys():
                self.disappeared[objectID] += 1

                # deregister if reached a max no.of consecutive 
                # frames where a given object has been marked as missing
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.dereigister(objectID)
            
            # return early as there are no centroids or tracking info
            # to update
            return self.objects
        
        # initialize an array of input centroids for the current frame
        inputCentroids = np.zeros((len(rects), 2), dtype = "int")

        # loop over the bounding box rectangles
        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            # use the bounding box coordinates to derive the centroid
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            inputCentroids[i] = (cX, cY)
        
        # if we are currently not tracking objects we take the input
        # centroid and register each one of them
        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i])
        
        else:
            # grab the set of object IDs and corresponding centroids
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())

            # compute the distance between each pair of object centroids and
            # input centroids
            D = dist.cdist(np.array(objectCentroids), inputCentroids)

            # perform matching
            rows = D.min(axis = 1).argsort()
            cols = D.argmin(axis = 1)[rows]

            # keeping track of which of the column and row indexes have
            # been already examined
            usedRows = set()
            usedCols = set()

            # loop over the combination of the (row, column) index tuples
            for (row, col) in zip(rows, cols):
                if row in usedRows or col in usedCols:
                    continue
                
                # grab the object ID for the current row, set its
                # new centroid and reset the disappeared counter
                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                self.disappeared[objectID] = 0

                # indicate we have examined each of row and column indexes
                usedRows.add(row)
                usedCols.add(col)
            
            # compute both the row and column index we have NOT yet examined
            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            # If no.of object centroids is equal or greater than no.of input
            # centroids we need to check and see if some objects have
            # potentially disappeared
            if(D.shape[0] >= D.shape[1]):
                # loop over the unused row indexes
                for row in unusedRows:
                    # grab the object ID for the corresponding row
                    # index and increment the disappeared counter
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1

                    # check to see if the no.of consecutive frames the 
                    # object has been marked "disappeared" for warrants
                    # deregistering the object
                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.dereigister(objectID)

            # otherwise if no.of input centroids is greater than the no.of
            # existing object centroids we need to register each new input
            # centroid as a trackable object
            else:
                for col in unusedCols:
                    self.register(inputCentroids[col])

        # return the set of trackable objects
        return self.objects 