# import the necssary packages
import numpy as np 

class Searcher:
    def __init__(self, index):
        # store our index of images
        self.index = index

    def search(self, queryFeatures):
        # intialise our dictionary of results
        results = {}
    
        # loop over the index
        for (k, features) in self.index.items():
            # compute the chi-squared distance between the 
            # features in our index and our query features using
            # the chi-squared distance which is normally used in the 
            # computer vision field to compute histograms
            d = self.chi2_distance(features, queryFeatures)

            # update the results dictionary so that key is the
            # current image ID in the index and the value is 
            # the distance we just compared, representing how 
            # similar the image in the index to our query
            results[k] = d
        
        # sort our results, so that smaller distances are at front of the list
        results = sorted([(v, k) for (k, v) in results.items()])

        # return our results
        return results

    def chi2_distance(self, histA, histB, eps=1e-10):
        # compute the chi-squared distance
        d = 0.5 * np.sum([((a-b) ** 2) / (a + b + eps) for (a,b) in zip(histA, histB)])

        # return the chi-squared distance
        return d