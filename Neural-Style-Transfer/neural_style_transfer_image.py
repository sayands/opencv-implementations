# import the necessary packages
import argparse
import imutils
import time
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required = True, help = "neural style transfer model")
ap.add_argument("-i", "--image", required = True, help = "input image to apply Neural Style Transfer to")
args = vars(ap.parse_args())

# load the neural style transfer model from disk
print("[INFO] loading neural style transfer model...")
net = cv2.dnn.readNetFromTorch(args["model"])

# load input image, resize and grab dimensions
image = cv2.imread(args["image"])
image = imutils.resize(image, width = 600)
(h, w) = image.shape[:2]

# construct a blob from image, set input and perform a forward pass
blob = cv2.dnn.blobFromImage(image, 1.0, (w,  h), (103.969, 116.779, 123.680), swapRB = False, crop = False)
net.setInput(blob)
start = time.time()
output = net.forward()
end = time.time()

# reshape the output tensor, add the mean subtraction and then swap the channel ordering
output = output.reshape((3, output.shape[2], output.shape[3]))
output[0] += 103.969
output[1] += 116.779
output[2] += 123.680
output /= 255.0
output = output.transpose(1, 2, 0)

# show relevant information and images
print("[INFO] neural style transfer took {:.4f} seconds".format(end - start))

# show the images
cv2.imshow("Input", image)
cv2.imshow("Output", output)
cv2.waitKey(0)