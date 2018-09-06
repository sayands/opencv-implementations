# import the necessary packages
import numpy as np 
import argparse
import imutils
import time
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required = True, help = "Path to deep learning segmentation model")
ap.add_argument("-c", "--classes", required = True, help = "Path to .txt file containing class labels")
ap.add_argument("-i", "--image", required= True, help = "Path to input image")
ap.add_argument("-l", "--colors", type= str, help = "Path to .txt file containing colors for labels")
ap.add_argument("-w", "--width", type=int, default = 500, help = "desired width(in pixels) of input image")
args = vars(ap.parse_args())

# load the class label names
CLASSES = open(args["classes"]).read().strip().split("\n")

# if a colors file was supplied, load it from disk
if args['colors']:
    COLORS = open(args["colors"]).read().strip().split("\n")
    COLORS = [np.array(c.split(",")).astype("int") for c in COLORS]
    COLORS = np.array(COLORS, dtype="uint8")

# otherwise, we need to randomly generate RGB colors for each class label
else:
    # initialize a list of colors to represent each class label in
    # the mask( starting with 'black' for the background/unlabeled
    # regions)
    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size = (len(CLASSES) -1, 3), dtype="uint8")
    COLORS = np.vstack([[0, 0, 0], COLORS]).astype("uint8")

#  initialize the legend visualisation
legend = np.zeros(((len(CLASSES) * 25) + 25, 300, 3), dtype="uint8")

# loop over the class names + colors
for (i, (className, color)) in enumerate(zip(CLASSES, COLORS)):
    # draw the class name + color on the legend
    color = [int(c) for c in color]
    cv2.putText(legend, className, (5, (i * 25) + 17), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.rectangle(legend, (100, (i * 25)), (300, (i * 25) + 25), tuple(color), -1)

# load our serialized model from the disk
print("[INFO] loading model...")
net = cv2.dnn.readNet(args["model"])

# load input image, resize it, construct a blob but keep in mind
# that original image dimensions on which ENet was trained was 1024 * 512
image = cv2.imread(args['image'])
image = imutils.resize(image, width = args["width"])
blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (1024, 512), 0, 
                            swapRB = True, crop = False)

# perform a forward pass using the segmentation model
net.setInput(blob)
start = time.time()
output = net.forward()
end = time.time()

# show the amount of time inference took
print("[INFO] inference took {:4f} seconds".format(end - start))

# infer the total no.of classes alongwith spatial dimensions of the mask
# image via the shape of the output array
(numClasses, height, width) = output.shape[1:4]

# our output class ID map will be num_classes * height * width in size
# take argmax to find the class label with the largest probability for each
# (x, y) coordinate in image
classMap = np.argmax(output[0], axis=0)

# given the class ID map, we can map each o the class IDs to its class Color
mask = COLORS[classMap]

mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation = cv2.INTER_NEAREST)
classMap = cv2.resize(classMap, (image.shape[1], image.shape[1]), interpolation = cv2.INTER_NEAREST)

# perform a weighted combination of input image with mask
output = ((0.4 * image) + (0.6 * mask)).astype("uint8")

# show output and input images
cv2.imshow("Legend", legend)
cv2.imshow("Input", image)
cv2.imshow("Output", output)
cv2.waitKey(0)
