# import the necessary packages
import numpy as np 
import argparse
import imutils
import cv2

# construct the argument and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--puzzle", required = True, help = "Path to the puzzle image")
ap.add_argument("-w", "--waldo", required = True, help = "Path to the waldo image")
args = vars(ap.parse_args())

# load the waldo and puzzle image
puzzle = cv2.imread(args["puzzle"])
waldo = cv2.imread(args["waldo"])
(waldoHeight, waldoWidth) = waldo.shape[:2]

# find the waldo in the puzzle
result = cv2.matchTemplate(puzzle, waldo, cv2.TM_CCOEFF)
(_, _, minLoc, maxLoc ) = cv2.minMaxLoc(result)

# the puzzle image
topLeft = maxLoc
botRight = (topLeft[0] + waldoWidth, topLeft[1] + waldoHeight)
roi = puzzle[topLeft[1] : botRight[1], topLeft[0] : botRight[0]]

# construct a darkened transparent 'layer' to darken everything
# in the puzzle except for waldo
mask = np.zeros(puzzle.shape, dtype="uint8")
puzzle = cv2.addWeighted(puzzle, 0.25, waldo, 0.75, 0)

# put the original waldo back in the image so that he is
# 'brighter' than rest of the image
puzzle[topLeft[1] : botRight[1], topLeft[0]:botRight[0]] = roi

# display the images
cv2.imshow('Puzzle', imutils.resize(puzzle, height = 650))
cv2.imshow("Waldo", Waldo)
cv2.waitKey(0)
