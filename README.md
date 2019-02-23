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
- ### Dataset 
    The dataset has the following structure, containing actor images collected using the Bing Image API (getData.py).
    
    
    ![screenshot from 2019-01-11 19-19-13](https://user-images.githubusercontent.com/31413064/51037313-dc50a200-15d5-11e9-94d1-45e94290ee52.png)
    
    
    As I'm using Spiderman 2, I've collected several pictures of the actors in the movie. Per actor I have ~15 images. More will do better, but this number seems to work fine.

- ### Training
  For every image in the dataset, we first get a square enclosing the the face in the image, then generate a 128d vector for that face, which is dumped to the 'encodings.pickle' file.
  
  We can either use CNN(slower, more accurate) or HOG(faster, less accurate) for the face detection process. Here I've used the face_recognition library, which gives me both the options.

- ### Face Recognition
  Consider an image, be it a still from the movie, or a frame of a video clip.
  First, we identify the faces in the image using the same method as above (CNN or HOG), generate an encoding for it(128d vector), and then compare it to our collected encodings. The actors with the most matched encodings is the actor in the image.

## Usage

Read the first few lines of the Python file involved to understand the parameters used in each case

- ### Making encodings
    ```
    python3 faceEncode.py --dataset dataset --encodings encodings.pickle -d hog
    ```

- ### Face Recognition in Image
  ```
  python3 faceRecImage.py -e encodings.pickle -i examples/ex6.png -d hog
  ```

- ### Face Recognition in Video File
  ```
  python3 faceRecVideoFile.py -e encodings.pickle -o output_vids/ex2.mp4 -y 0 -d hog --input input_vids/ex2.mp4
  ```
  Outputs a video with the faces marked.

- ### Face Recognition in Video Stream from Webcam
  ```
  python3 faceRecVideo.py -e encodings.pickle -o output_vids/ex1.avi -y 0 -d hog
  ```
- ### Getting Image Data
  ```
  python3 getData.py --query "tobey macguire" --output dataset/tobey_macquire
  ```
  Will fetch images from Bing Image Search and save in the mentioned directory (Max 50).

#### Feel free to fork the repository and use it on your own movies, maybe expand the dataset and make it a general software for any given movie :)

