# for every folder in bigdata directory, make sure it only has only 25 pictures, delete rest of the files

import os
import shutil
import random

def make_data_small():
    bigdata_dir = "bigdata"
    # iterate over all the folders in bigdata
    for folder in os.listdir(bigdata_dir):
        # keep only 25 images in each folder
        folder_path = os.path.join(bigdata_dir, folder)
        if os.path.isdir(folder_path):
            # get all the files in the folder
            files = os.listdir(folder_path)
            # shuffle the files
            random.shuffle(files)
            # keep only 25 files
            files = files[:25]
            # delete all the files except the 25 files
            for file in os.listdir(folder_path):
                if file not in files:
                    os.remove(os.path.join(folder_path, file))

if __name__ == "__main__":
    make_data_small()
