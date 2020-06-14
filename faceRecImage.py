import face_recognition
import argparse
import pickle
import cv2
from sklearn.neighbors import KDTree
import numpy as np

NORM_DIST_TOLERANCE = 0.6
no_nn = 5

def find_best_match_within_tolerance(candidates,names,tolerance): 
    zipped_dist_names = np.dstack(candidates)
    best_candidates = []
    for candidates in zipped_dist_names: 
        count = {}
        best_candidate = "Unknown"
        filtered_candidates = [int(ind) for dist,ind in candidates if dist <= tolerance]
        if len(filtered_candidates) != 0:
            for ind in filtered_candidates:
                count[names[ind]] = count.get(names[ind],0) + 1
            best_candidate = max(count,key = count.get)
        best_candidates.append(best_candidate)
    return best_candidates

def fast_face_match_knn(data, query_encodings,tolerance):
    kdtree = data["encodings"]
    results = kdtree.query(query_encodings,no_nn)
    return find_best_match_within_tolerance(results,data["names"],tolerance)


def linear_search(data,query_encodings):
    names = []
    # loop over the facial embeddings
    for encoding in query_encodings:
    # attempt to match each face in the input image to our known
    # encodings
        matches = face_recognition.compare_faces(data["encodings"],
            encoding)
        name = "Unknown"

        # check to see if we have found a match
        if True in matches:
            # find the indexes of all matched faces then initialize a
            # dictionary to count the total number of times each face
            # was matched
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}

            # loop over the matched indexes and maintain a count for
            # each recognized face
            for i in matchedIdxs:
                name = data["names"][i]
                counts[name] = counts.get(name, 0) + 1

            # determine the recognized face with the largest number of
            # votes (note: in the event of an unlikely tie Python will
            # select first entry in the dictionary)
            name = max(counts, key=counts.get)

        # update the list of names
        names.append(name)
    return names

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--encodings", required=True,
    help="path to serialized db of facial encodings")
ap.add_argument("-i", "--image", required=True,
    help="path to input image")
ap.add_argument("-d", "--detection-method", type=str, default="cnn",
    help="face detection model to use: either `hog` or `cnn`")
ap.add_argument("-fnn","--fast-nn",action="store_true")
args = vars(ap.parse_args())
print(args)

# load the known faces and embeddings
print("[INFO] loading encodings...")
data = pickle.loads(open(args["encodings"], "rb").read())
 
# load and convert the image from BGR color (which OpenCV uses) 
# to RGB color (which face_recognition uses)
image = cv2.imread(args["image"])
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
 
# detect the (x, y)-coordinates of the bounding boxes corresponding
# to each face in the input image, then compute the facial embeddings
# for each face
print("[INFO] recognizing faces...")
boxes = face_recognition.face_locations(rgb,
    model=args["detection_method"])
encodings = face_recognition.face_encodings(rgb, boxes)


names = []
if "fast_nn" in args:
    names = fast_face_match_knn(data,encodings,NORM_DIST_TOLERANCE)
else:
    names = linear_search(data,encodings)
print("names : ",names)

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