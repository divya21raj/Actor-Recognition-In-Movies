import face_recognition
import numpy as np
from sklearn.neighbors import KDTree

import constants


def _linear_search(known_encodings, query_encodings, known_names):
    names = []
    # loop over the facial embeddings
    for encoding in query_encodings:
        # attempt to match each face in the input image to our known
        # encodings
        matches = face_recognition.compare_faces(known_encodings, encoding)
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
                name = known_names[i]
                counts[name] = counts.get(name, 0) + 1

            # determine the recognized face with the largest number of
            # votes (note: in the event of an unlikely tie Python will
            # select first entry in the dictionary)
            name = max(counts, key=counts.get)

        # update the list of names
        names.append(name)
    return names


def _find_best_match_within_tolerance(candidates, names, tolerance):
    zipped_dist_names = np.dstack(candidates)
    best_candidates = []

    for candidates in zipped_dist_names:
        count = {}
        best_candidate = constants.ID_UNKNOWN
        filtered_candidates = [int(ind) for dist, ind in candidates if dist <= tolerance]

        if len(filtered_candidates) != 0:
            for ind in filtered_candidates:
                count[names[ind]] = count.get(names[ind], 0) + 1
            best_candidate = max(count, key=count.get)

        best_candidates.append(best_candidate)
    return best_candidates


# find the best match for the given set of query encodings with given tolerance for distance value
# returns then names of matched actors or constants.ID_UNKNOWN in case of no valid match with tolerance

# find the k nearest neighbors using precomputed kdtree of training encodings
# return the names of most face for query or constants.ID_UNKNOWN in case of no valid match
def _fast_face_match_knn(known_encodings, query_encodings, known_names, tolerance, k):
    kdtree = known_encodings
    results = kdtree.query(query_encodings, k)
    return _find_best_match_within_tolerance(results, known_names, tolerance)


class FaceRec:
    """
    Class containing methods for recognizing faces in a given image.
    """

    @staticmethod
    def getAllFacesInImage(image_rgb, detection_method, use_fastnn, known_encodings,
                           known_encodings_structure, known_names):
        """
        Method which detects all the faces in a particular image
        :param image_rgb: image in rgb color
        :param detection_method: what detection method to use for detecting the face (hog or cnn)
        :param use_fastnn: Whether to use the kdtree implementation for searching names
        :param known_encodings: Encoding representing the dataset
        :param known_encodings_structure: Structure of the known_encoding provided (linear or kdtree)
        :param known_names: Names from the dataset, corresponding with the known_encodings
        """
        # detect the (x, y)-coordinates of the bounding boxes corresponding
        # to each face in the input image, then compute the facial embeddings
        # for each face
        boxes = face_recognition.face_locations(image_rgb, model=detection_method)
        encodings = face_recognition.face_encodings(image_rgb, boxes)

        names = []

        if encodings:
            if use_fastnn or known_encodings_structure == constants.ENC_KDTREE:
                # check if kdtree is to be recomputed or not
                if known_encodings_structure != constants.ENC_KDTREE:
                    known_encodings = KDTree(np.asarray(encodings), leaf_size=constants.LEAF_SIZE_KDTREE)
                    encoding_structure = constants.ENC_KDTREE

                names = _fast_face_match_knn(known_encodings, encodings, known_names, constants.NORM_DIST_TOLERANCE,
                                             constants.K_NN)
            else:
                names = _linear_search(known_encodings, encodings, known_names)

        return names, boxes
