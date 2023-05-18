# Actor Recognition In Movie Clips and Images
<img src="https://github.com/divya21raj/Actor-Recognition-In-Movies/assets/31413064/7407cf08-17c8-4d90-a385-b3aded2739dd" width="539" height="262">

<img src="https://github.com/divya21raj/Actor-Recognition-In-Movies/assets/31413064/037b0201-58a0-4cc1-8954-07fa5d27ea18" width="539" height="300">

<img src="https://github.com/divya21raj/Actor-Recognition-In-Movies/assets/31413064/b2993f49-f8c6-4033-b205-d03dabefd07b" width="539" height="300">

<img src="https://github.com/divya21raj/Actor-Recognition-In-Movies/assets/31413064/7e6805c2-7e7a-40d5-89b4-d2523a38e7fd" width="539" height="304">

![ex1](https://user-images.githubusercontent.com/31413064/51027993-43f7f480-15b8-11e9-809a-f711c59aac8a.gif)

Recognizing actors/celebs in a clip or image from **any** media, using DeepLearning with Python.
Can use either CNN or HOG for face detection and then compare the face with our dataset of faces.

I have used the mostly comprehensive dataset available [here](https://github.com/prateekmehta59/Celebrity-Face-Recognition-Dataset). It is only updated with celebrity faces till 2021, so we might need to update it further if required.

Tons of help from ageitgey's [face_recognition](https://github.com/ageitgey/face_recognition) library.

Inspired by [this](https://www.pyimagesearch.com/2018/06/18/face-recognition-with-opencv-python-and-deep-learning/) wonderful article.

## Process

- ### Setup
    Install `cmake`, as it is required for the dlib library.
    For linux, run `sudo apt-get install cmake`
    For Windows, download the installer from [here](https://cmake.org/download/)
    For macOS, run `brew install cmake`

    Also, install `pipenv` for managing the virtual environment.
    For linux, run `sudo pip install pipenv`
    For Windows, run `pip install pipenv`
    For macOS, run `brew install pipenv`

    Then, run `pipenv shell` to activate the virtual environment.
    Finally, run `pipenv install` to install all the dependencies.
    
    Note: the `face-recognition` library does not officially support Windows, but it still might work, as it says in its [README](https://github.com/ageitgey/face_recognition)
    
- ### Dataset 
    The dataset has the following structure.
    
    ![screenshot from 2019-01-11 19-19-13](https://user-images.githubusercontent.com/31413064/51037313-dc50a200-15d5-11e9-94d1-45e94290ee52.png)
    
    For my implementation, each actor has 25 images. More will do better, but this number seems to work fine.

- ### Training
  For every image in the dataset, we first get a square enclosing the face in the image, then generate a 128d vector for that face, which is dumped to the 'encodings.pickle' file.
  
  We can either use CNN(slower, more accurate) or HOG(faster, less accurate) for the face detection process. Here I've used the face_recognition library, which gives me both the options.

  For a big dataset, techniques like MapReduce or Spark can be used to parallelize the process over a cluster of machines.

  Moreover, use the `-fnn` flag in case you want to use the KDTree method for searching, which is much faster than the linear search.

- ### Face Recognition
  Consider an image, be it a still from the movie, or a frame of a video clip.
  First, we identify the faces in the image using the same method as above (CNN or HOG), generate an encoding for it(128d vector), and then compare it with our collected encodings. The actors with the most matched encodings is the actor in the image.

  This search can either be linear, or using a KDTree. I've used the KDTree method, which is much faster. This can be done by passing the `-fnn` flag to the python file.
## Usage

Read the first few lines of the Python file involved to understand the parameters used in each case

- ### Making encodings
    ```
    python faceEncode.py --dataset dataset/actors --encodings encodings/encodings.pickle -d hog -c 8
    ```

    `-c` flag is the number of cores to use for parallel processing.

    Can also use the `-fnn` flag to later use the KDTree method for searching. 

- ### Face Recognition in Image
  ```
  python faceRecImage.py -e encodings.pickle -i examples/ex6.png -d hog -o out/
  ```

  Use the `-fnn` flag to use the KDTree method for searching.

- ### Face Recognition in Video File
  ```
  python faceRecVideoFile.py -e encodings/encodings.pickle -i input_vids/ex2.mp4 -o output_vids/ex2.avi -y 0 -d hog 
  ```
  Outputs a video with the faces marked.

#### Feel free to fork the repository and use it on your own dataset. The `encodings/encodings/encodings_fnn_big.pickle` file is already trained on a big dataset (1100 celebs with 25 images each), so you can use it directly for face recognition.

