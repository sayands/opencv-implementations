# import the necessary packages
from pyimagesearch import imutils
import numpy as np
import argparse
import cv2

# construct the argument and parse the argument
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help = "path to the (optional) video file")
args = vars(ap.parse_args())

# define the upper and lower boundaries of the HSV pixel
# intensities to be considered 'skin'

lower = np.array([0, 30, 60], dtype = 'uint8')
upper = np.array([20, 255, 255], dtype="uint8")

# if a video path was not supplied, grab a reference to the gray
if not args.get('video', False):
    camera = cv2.VideoCapture(0)

# otherwise, load the video
else:
    camera = cv2.VideoCapture(args["video"])

# keep looping over the frames in the video
while True:
    # grab the current frame
    (grabbed, frame) = camera.read()

    # if we are viewing a video and we didnt grab frame then we have reached the end of the video
    if args.get('video') and not grabbed:
        break
    
    # resize the frame, convert it to the HSV color space,
    # and determine the HSV pixel intensities that fall into
    # the specified upper and lower boundaries
    frame = imutils.resize(frame, width = 400)
    converted = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    skinMask = cv2.inRange(converted, lower, upper)

    # apply a series of erosions and dilations to the mask
    # using an elliptical kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    skinMask = cv2.erode(skinMask, kernel, iterations = 2)
    skinMask = cv2.dilate(skinMask, kernel, iterations = 2)

    # blur the mask to help remove noise, then apply
    # mask to the frame
    skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)
    skin = cv2.bitwise_and(frame, frame, mask = skinMask)

    # show the skin in the image along with the mask
    cv2.imshow("images", np.hstack([frame, skin]))

    # if the 's' key is pressed, stop the loop
    if cv2.waitKey(1) & 0xFF == ord('s'):
        break

# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()