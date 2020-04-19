from requests import exceptions
import argparse
import requests
import cv2
import os

# Argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-q", "--query", required=True,
    help="search query to search Bing Image API for")
ap.add_argument("-o", "--output", required=True,
    help="path to output directory of images")
args = vars(ap.parse_args())

# GLOBALS
API_KEY = "b11b37d626f24592a33af7c24a76229b"
MAX_RESULTS = 250
GROUP_SIZE = 50
URL = "https://api.cognitive.microsoft.com/bing/v7.0/images/search"

# List of Exceptions
EXCEPTIONS = set([IOError, FileNotFoundError,
    exceptions.RequestException, exceptions.HTTPError,
    exceptions.ConnectionError, exceptions.Timeout])

# Search starts here
term = args["query"]
headers = {"Ocp-Apim-Subscription-Key" : API_KEY}
params = {"q": term, "offset": 0, "count": GROUP_SIZE}
 
# Make the search
print("[INFO] searching Bing API for '{}'".format(term))
search = requests.get(URL, headers=headers, params=params)
search.raise_for_status()
 
# grab the results from the search, including the total number of
# estimated results returned by the Bing API
results = search.json()
estNumResults = min(results["totalEstimatedMatches"], MAX_RESULTS)
print("[INFO] {} total results for '{}'".format(estNumResults,
    term))


total = 0
# loop over the estimated number of results in `GROUP_SIZE` groups
for offset in range(0, estNumResults, GROUP_SIZE):
    # update the search parameters using the current offset, then
    # make the request to fetch the results
    print("[INFO] making request for group {}-{} of {}...".format(
        offset, offset + GROUP_SIZE, estNumResults))
    params["offset"] = offset
    search = requests.get(URL, headers=headers, params=params)
    search.raise_for_status()
    results = search.json()
    print("[INFO] saving images for group {}-{} of {}...".format(
        offset, offset + GROUP_SIZE, estNumResults))

# loop over the results
    for v in results["value"]:
        # try to download the image
        try:
            # make a request to download the image
            print("[INFO] fetching: {}".format(v["contentUrl"]))
            r = requests.get(v["contentUrl"], timeout=30)
 
            # build the path to the output image
            ext = v["contentUrl"][v["contentUrl"].rfind("."):]
            p = os.path.sep.join([args["output"], "{}{}".format(
                str(total).zfill(8), ext)])
 
            # write the image to disk
            f = open(p, "wb")
            f.write(r.content)
            f.close()
 
        # catch any errors that would not unable us to download the
        # image
        except Exception as e:
            # check to see if our exception is in our list of
            # exceptions to check for
            if type(e) in EXCEPTIONS:
                print("[INFO] skipping: {}".format(v["contentUrl"]))
                continue

        # try to load the image from disk
        image = cv2.imread(p)
 
        # if the image is `None` then we could not properly load the
        # image from disk (so it should be ignored)
        if image is None:
            print("[INFO] deleting: {}".format(p))
            os.remove(p)
            continue
 
        # update the counter
        total += 1

