import argparse
import pickle

import cv2

import constants
from faceRec import FaceRec

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--encodings", required=True,
                help="path to serialized db of facial encodings")
ap.add_argument("-i", "--image", required=True,
                help="path to input image")
ap.add_argument("-d", "--detection-method", type=str, default="cnn",
                help="face detection model to use: either `hog` or `cnn`")
ap.add_argument("-fnn", "--fast-nn", action="store_true")
args = vars(ap.parse_args())
print(args)

# load the known faces and embeddings
print("[INFO] loading encodings...")
data = pickle.loads(open(args["encodings"], "rb").read())

# load and convert the image from BGR color (which OpenCV uses) 
# to RGB color (which face_recognition uses)
image = cv2.imread(args["image"])
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

faceRec = FaceRec()

print("[INFO] recognizing faces...")
# based on user args select fast kdtree based nn or linear search
names, boxes = faceRec.getAllFacesInImage(image_rgb, args["detection_method"], args["fast_nn"],
                                          data[constants.KNOWN_ENCODINGS], data[constants.ENCODING_STRUCTURE],
                                          data[constants.KNOWN_NAMES])
print("names : ", names)

# loop over the recognized faces
for ((top, right, bottom, left), name) in zip(boxes, names):
    # draw the predicted face name on the image
    cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
    y = top - 15 if top - 15 > 15 else top + 15
    cv2.putText(image, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
                0.75, (0, 255, 0), 2)

# show the output image
cv2.imshow("Image", image)
cv2.waitKey(0)
