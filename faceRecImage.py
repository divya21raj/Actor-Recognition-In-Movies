import face_recognition
import argparse
import pickle
import cv2
from sklearn.neighbors import KDTree
import numpy as np

import constants

def linear_search(data,query_encodings):
    names = []
    # loop over the facial embeddings
    for encoding in query_encodings:
        # attempt to match each face in the input image to our known
        # encodings
        matches = face_recognition.compare_faces(data[constants.ENCODINGS],
            encoding)
        name = constants.ID_UNKNOWN

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
                name = data[constants.NAMES][i]
                counts[name] = counts.get(name, 0) + 1

            # determine the recognized face with the largest number of
            # votes (note: in the event of an unlikely tie Python will
            # select first entry in the dictionary)
            name = max(counts, key=counts.get)

        # update the list of names
        names.append(name)
    return names

# find the best match for the given set of query encodings with given tolerance for distance value
# returns then names of matched actors or constants.ID_UNKNOWN in case of no valid match with tolerance
def find_best_match_within_tolerance(candidates, names, tolerance): 
    zipped_dist_names = np.dstack(candidates)
    best_candidates = []
    
    for candidates in zipped_dist_names: 
        count = {}
        best_candidate = constants.ID_UNKNOWN
        filtered_candidates = [int(ind) for dist, ind in candidates if dist <= tolerance]
        
        if len(filtered_candidates) != 0:
            for ind in filtered_candidates:
                count[names[ind]] = count.get(names[ind], 0) + 1
            best_candidate = max(count, key = count.get)
        
        best_candidates.append(best_candidate)
    return best_candidates

# find the k nearest neighbors using precomputed kdtree of training encodings
# return the names of most face for query or constants.ID_UNKNOWN in case of no valid match
def fast_face_match_knn(data, query_encodings, tolerance, k):
    kdtree = data[constants.ENCODINGS]
    results = kdtree.query(query_encodings, k)
    return find_best_match_within_tolerance(results, data[constants.NAMES], tolerance)


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

# based on user args select fast kdtree based nn or linear seach
names = []
if args["fast_nn"] or data[constants.ENCODING_STRUCTURE] == constants.ENC_KDTREE:
    # check if kdtree is to be recomputed or not
    if data[constants.ENCODING_STRUCTURE] != constants.ENC_KDTREE :
        data[constants.ENCODINGS] = KDTree(np.asarray(data[constants.ENCODINGS]),
                                           leaf_size=constants.LEAF_SIZE_KDTREE)
        data[constants.ENCODING_STRUCTURE] = constants.ENC_KDTREE
    
    names = fast_face_match_knn(data, encodings, constants.NORM_DIST_TOLERANCE, constants.K_NN)
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
