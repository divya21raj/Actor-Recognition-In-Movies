# Actor Recognition In Movie Clips and Images
![ex1](https://user-images.githubusercontent.com/31413064/51027993-43f7f480-15b8-11e9-809a-f711c59aac8a.gif)


![ex1_540](https://user-images.githubusercontent.com/31413064/51028169-c5e81d80-15b8-11e9-9c15-d12ce4905027.png)


![ex2_540](https://user-images.githubusercontent.com/31413064/51028202-d7312a00-15b8-11e9-9c6e-948bc385967d.png)

Recognizing actors in a movie clip or image, using DeepLearning and Python.
Can use either CNN or HOG for face detection and then compare the face with our dataset of faces.

Here I've used **Spiderman 2** (2004), as an example. It can work with any piece of media, given the right dataset.

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
    The dataset has the following structure, containing actor images collected using the Bing Image API (getData.py).
    
    
    ![screenshot from 2019-01-11 19-19-13](https://user-images.githubusercontent.com/31413064/51037313-dc50a200-15d5-11e9-94d1-45e94290ee52.png)
    
    
    As I'm using Spiderman 2, I've collected several pictures of the actors in the movie. Per actor I have ~15 images. More will do better, but this number seems to work fine.

- ### Training
  For every image in the dataset, we first get a square enclosing the face in the image, then generate a 128d vector for that face, which is dumped to the 'encodings.pickle' file.
  
  We can either use CNN(slower, more accurate) or HOG(faster, less accurate) for the face detection process. Here I've used the face_recognition library, which gives me both the options.

- ### Face Recognition
  Consider an image, be it a still from the movie, or a frame of a video clip.
  First, we identify the faces in the image using the same method as above (CNN or HOG), generate an encoding for it(128d vector), and then compare it with our collected encodings. The actors with the most matched encodings is the actor in the image.

## Usage

Read the first few lines of the Python file involved to understand the parameters used in each case

- ### Making encodings
    ```
    python faceEncode.py --dataset dataset/actors --encodings encodings/encodings.pickle -d hog
    ```

- ### Face Recognition in Image
  ```
  python faceRecImage.py -e encodings.pickle -i examples/ex6.png -d hog
  ```

- ### Face Recognition in Video File
  ```
  python faceRecVideoFile.py -e encodings/encodings.pickle -i input_vids/ex2.mp4 -o output_vids/ex2.mp4 -y 0 -d hog 
  ```
  Outputs a video with the faces marked.

- ### Getting Image Data
  ```
  python getData.py --query "tobey maguire" --output dataset/tobey_maguire
  ```
  Will fetch images from Bing Image Search and save in the mentioned directory (Max 50).
  Make sure to get your own Bing search API key from [here](https://azure.microsoft.com/en-us/try/cognitive-services/?api=bing-image-search-api) and fill it up in the code.

#### Feel free to fork the repository and use it on your own movies, maybe expand the dataset and make it a general software for any given movie :)

