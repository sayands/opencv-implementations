# Import the neccessary packages
from imutils import face_utils
from imutils import paths 
import numpy as np 
import argparse 
import imutils 
import shutil 
import json 
import dlib 
import cv2 
import os 
import sys

def overlay_image(bg, fg, fgMask, coords):
    # grab the foreground spatial dimensions
    (sH, sW) = fg.shape[:2]
    (x, y) = coords

    overlay = np.zeros(bg.shape, dtype = "uint8")
    overlay[y:y + sH, x: x + sW] = fg 

    alpha = np.zeros(bg.shape[:2], dtype = "uint8")
    alpha[y:y + sH, x: x + sW] = fgMask
    alpha = np.dstack([alpha] * 3)

    # perform alpha blending to merge background, foreground and alpha channel
    output = alpha_blend(overlay, bg, alpha)

    return output 

def alpha_blend(fg, bg, alpha):
    fg = fg.astype("float")
    bg = bg.astype("float")
    alpha = alpha.astype("float") / 255 

    # perform alpha blending
    fg = cv2.multiply(alpha, fg)
    bg = cv2.multiply(1 - alpha, bg)

    # add the foreground and background to obtain the final output image
    output = cv2.add(fg, bg)

    # return the output image
    return output.astype("uint8")

def create_gif(inputPath, outputPath, delay, finalDelay, loop):
    # grab all image paths in the input directory
    imagePaths = sorted(list(paths.list_images(inputPath)))

    # remove the last image in the list
    lastPath = imagePaths[-1]
    imagePaths = imagePaths[:-1]

    cmd = "convert -delay {} {} -delay {} {} -loop {} {}".format(delay, " ".join(imagePaths), finalDelay, lastPath, loop, outputPath)
    os.system(cmd)
    

