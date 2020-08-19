from imutils import paths
import face_recognition
import argparse
import pickle
import cv2
import os
from sklearn.neighbors import KDTree
import numpy as np
import constants
import multiprocessing as mp
import functools


def _encode_faces(imagePath, detection_method):
    known_encodings = []
    known_names = []
    try:
        name = imagePath.split(os.path.sep)[-2]

        # load the input image and convert it from BGR (OpenCV ordering)
        # to dlib ordering (RGB)
        image = cv2.imread(imagePath)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # detect the (x, y)-coordinates of the bounding boxes
        # corresponding to each face in the input image
        boxes = face_recognition.face_locations(rgb,
                                                model=detection_method)

        # compute the facial embedding for the face
        encodings = face_recognition.face_encodings(rgb, boxes)

        # loop over the encodings
        for encoding in encodings:
            # add each encoding + name to our set of known names and
            # encodings
            known_encodings.append(encoding)
            known_names.append(name)
        return known_encodings, known_names
    except Exception as ex:
        print("exception while encoding image %s : %s" % (imagePath, ex))
        return [], []


def _process_images():
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--dataset", required=True,
                    help="path to input directory of faces + images")
    ap.add_argument("-e", "--encodings", required=True,
                    help="path to serialized db of facial encodings")
    ap.add_argument("-d", "--detection-method", type=str, default="cnn",
                    help="face detection model to use: either `hog` or `cnn`")
    ap.add_argument("-fnn", "--fast-nn", action="store_true")
    ap.add_argument("-c", "--cores", required=False, type=int, default=1,
                    help="no of cores to run on, will decide the parallelism")
    args = vars(ap.parse_args())

    # grab the paths to the input images in our dataset
    print("[INFO] quantifying faces...")
    image_paths = list(paths.list_images(args["dataset"]))
    known_encodings = []
    known_names = []
    mp_batch_size = args["cores"] * 10

    encode_images_with_detection_method = functools.partial(_encode_faces,
                                                            detection_method=args["detection_method"])

    # loop over the image paths using batching for multiprocessing
    for i in range(0, len(image_paths), mp_batch_size):
        image_batch = image_paths[i:i + mp_batch_size]
        # using pool for parallelism
        with mp.Pool(args["cores"] * 2) as pool:
            encodings_names_list = pool.map(encode_images_with_detection_method, image_batch)
        encodings_names_list = filter(lambda t: len(t[0]) > 0 and len(t[1]) > 0, encodings_names_list)
        for (encodings, names) in encodings_names_list:
            known_encodings.extend(encodings)
            known_names.extend(names)
        print("finished encoding {%d}/{%d} images" % (min(len(image_paths), i + mp_batch_size), len(image_paths)))

    # dump the facial encodings + names to disk
    print("[INFO] serializing encodings...")
    # select encoding as kdtree or list based on user args
    encoding_structure = constants.ENC_LIST

    if args["fast_nn"]:
        encoding_structure = constants.ENC_KDTREE
        known_encodings = KDTree(np.asarray(known_encodings), leaf_size=constants.LEAF_SIZE_KDTREE)

    data = {constants.KNOWN_ENCODINGS: known_encodings,
            constants.KNOWN_NAMES: known_names,
            constants.ENCODING_STRUCTURE: encoding_structure
            }

    f = open(args["encodings"], "wb")
    f.write(pickle.dumps(data))
    f.close()


if __name__ == "__main__":
    _process_images()
