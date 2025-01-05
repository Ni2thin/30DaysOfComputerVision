# 30 Days of Computer Vision 🚀

![Last Commit](https://img.shields.io/github/last-commit/Ni2thin/30DaysOfComputerVision)
![Repo Size](https://img.shields.io/github/repo-size/Ni2thin/30DaysOfComputerVision)

Like the rising sun, every new day brings an opportunity to shine brighter. With patience, determination, and resilience, we will paint our path in the world of computer vision. **七転び八起き** (Nanakorobi yaoki) – "Fall seven times, stand up eight."



## Resources and Progress 📚

| Books & Resources                                   | Completion Status | 
|-----------------------------------------------------|-------------------|
| [Machine Learning Specialization on Coursera](https://www.coursera.org/specializations/machine-learning-introduction) | 📈        |      
| [Hands-On Machine Learning with Scikit-Learn and TensorFlow](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/) |   📈   |      
| [Computer Vision YouTube Playlist](https://www.youtube.com/watch?v=HiTw5KFw7ic&list=PLb49csYFtO2HAdNGChGzohFJGnJnXBOqd) |  ![Playlist Thumbnail](https://img.youtube.com/vi/HiTw5KFw7ic/hqdefault.jpg) |

## Progress Tracker

| Day  | Date         | Topics                                               | Resources                                          |
|------|--------------|------------------------------------------------------|----------------------------------------------------|
| Day 1 | 27-12-2024  | OpenCV tutorial for beginners                        | [OpenCV tutorial](https://www.youtube.com/watch?v=eDIj5LuIL4A&list=PLb49csYFtO2HAdNGChGzohFJGnJnXBOqd&index=2)     |
| Day 2 | 28-12-2024  | Detecting color with Python and OpenCV               | [Detecting color](https://www.youtube.com/watch?v=aFNDh5k3SjU&list=PLb49csYFtO2HAdNGChGzohFJGnJnXBOqd&index=4)                   |
| Day 3 | 29-12-2024  | Face detection and blurring with Python              | [Face detection and blurring](https://www.youtube.com/watch?v=DRMBqhrfxXg&list=PLb49csYFtO2HAdNGChGzohFJGnJnXBOqd&index=4)       |
| Day 4 | 30-12-2024  | Text detection with Python(Tesseract)             | [Text detection with Python](https://www.youtube.com/watch?v=CcC3h0waQ6I&list=PLb49csYFtO2HAdNGChGzohFJGnJnXBOqd&index=5)        |
| Day 5 | 31-12-2024  | Image classification with Python and OpenCV          | [Image classification with Python](https://www.youtube.com/watch?v=il8dMDlXrIE&list=PLb49csYFtO2HAdNGChGzohFJGnJnXBOqd&index=6&pp=iAQB)  |
| Day 6 | 01-01-2025  | Emotion detection with Python, OpenCV, and others    | [Emotion detection](https://www.youtube.com/watch?v=h0LoewzGzhc&list=PLb49csYFtO2HAdNGChGzohFJGnJnXBOqd&index=8)                 |
| Day 7 | 02-01-2025  | Image classification + feature extraction           | [Image classification + feature extraction](https://youtu.be/oEKg_jiV1Ng?feature=shared) |
| Day 8 | 03-01-2025  | Sign language detection with Python and OpenCV       | [Sign language detection](https://youtu.be/MJCSjXepaAM?feature=shared)           |
| Day 9 | 04-01-2025  | Image classification WEB APP with Python & Flask     | [Image classification web app](https://youtu.be/n_eMARPqBZI?feature=shared)      |
| Day 10 | 05-01-2025 | AWS Rekognition tutorial (Object detection)      | [AWS Rekognition](link)                   |
| Day 11 | 06-01-2025 | Yolov8 object tracking 100% native                   | [Yolov8 object tracking](link)            |
| Day 12 | 07-01-2025 | Image segmentation with Yolov8 custom dataset       | [Image segmentation with Yolov8](link)    |
| Day 13 | 08-01-2025 | Train pose detection Yolov8 on custom dataset        | [Train pose detection Yolov8](link)       |
| Day 14 | 09-01-2025 | Parking spot detection and counter                   | [Parking spot detection](link)            |
| Day 15 | 10-01-2025 | Train Yolov10 object detection custom dataset        | [Train Yolov10 object detection](link)    |
| Day 16 | 11-01-2025 | End to end pipeline real world computer vision       | [End to end pipeline](link)               |
| Day 17 | 12-01-2025 | Image processing API with AWS API Gateway            | [Image processing API](link)              |
| Day 18 | 13-01-2025 | How much data you need to train a computer vision model | [How much data to train cv](link)                 |
| Day 19 | 14-01-2025 | Real world application of computer vision           | [Real world application](link)            |
| Day 20 | 15-01-2025 | Train Detectron2 object detection custom dataset     | [Train Detectron2](link)                  |
| Day 21 | 16-01-2025 | Face recognition on your webcam with Python & OpenCV | [Face recognition](link)                  |
| Day 22 | 17-01-2025 | Face attendance + face recognition with OpenCV      | [Face attendance](link)                   |
| Day 23 | 18-01-2025 | Machine learning with AWS practical projects        | [ML with AWS](link)                       |
| Day 24 | 19-01-2025 | Chat with an image (LangChain custom)                | [Chat with an image](link)                |
| Day 25 | 20-01-2025 | Image generation with Python (Train Stable Diffusion) | [Image generation](link)                  |
| Day 26 | 21-01-2025 | Face recognition + liveness detection               | [Face recognition + liveness](link)       |
| Day 27 | 22-01-2025 | Face recognition and face matching with Python      | [Face matching](link)                     |
| Day 28 | 23-01-2025 | Machine learning web app with Python, Flask, and others | [ML web app](link)                     |
| Day 29 | 24-01-2025 | Object detection on Raspberry Pi USB camera         | [Object detection on Raspberry Pi](link)  |
| Day 30 | 25-01-2025 | Image generation with Python & Stable Diffusion     | [Image generation](link)                  |

---

# Day 1: Getting started with OpenCV

## Key Concepts

- **Images in CV2**: Images are stored as **NumPy arrays** in OpenCV.
- **Image Shape**: The shape of an image consists of **height, width, and channels** (BGR format).
- **Color Format**: 
  - OpenCV uses **BGR** (Blue, Green, Red) while libraries like Matplotlib use **RGB**.
  - To convert from BGR to RGB, use:
    ```
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    ```
- [Introduction to CV2 ](https://github.com/Ni2thin/30DaysOfComputerVision/tree/main/CV2%20Intro)   

## Image Manipulation

- **Shape and Resize**: Use `cv2.resize()` to change the dimensions of an image.
- **Cropping**: Crop images by slicing the NumPy array.

## Color Space Conversions

- Convert between color spaces using functions like:
  - `cv2.cvtColor(image, cv2.COLOR_BGR2RGB)` for BGR to RGB
  - `cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)` for BGR to Grayscale
  - `cv2.cvtColor(image, cv2.COLOR_BGR2HSV)` for BGR to HSV

## Blurring Techniques

- Use various blurring methods such as:
  - `cv2.blur()`
  - `cv2.medianBlur()`
  - `cv2.GaussianBlur()`

## Thresholding Methods

- Apply different thresholding techniques for segmentation:
  - **Global Thresholding**
  - **Adaptive Thresholding**
  - Converts images from normal to binary.

## Edge Detection Techniques

- Utilize edge detection algorithms like:
  - `cv2.Sobel()`
  - `cv2.Canny()`
  - `cv2.Laplacian()`
- Morphological operations such as dilation and erosion can also be applied.

## Drawing Functions

- Draw shapes on images using functions like:
  - `cv2.line()`
  - `cv2.rectangle()`
  - `cv2.circle()`

## Contours

### Features:
- **Grayscale Conversion**: Converts the input image to grayscale for easier processing.
- **Thresholding**: Applies binary inverse thresholding to create a binary image suitable for contour detection.
- **Contour Extraction**: Uses OpenCV's `findContours` to extract object contours from the thresholded image.
- **Bounding Boxes**: Calculates and draws bounding rectangles around contours exceeding a specified area threshold.
- **Dynamic Visualization**: Displays the original image with bounding boxes and the thresholded binary image for analysis.

  ---
  
# Day 2: Detecting color with Python and OpenCV 

This project uses OpenCV to detect and track objects of a specific color in real-time using a webcam feed. It processes frames to isolate the target color, highlights detected objects with bounding boxes, and displays the result. [color detection](https://github.com/Ni2thin/30DaysOfComputerVision/blob/main/Projects/color-detection.py)

## Features

- Dynamic HSV Range:
Adapts HSV thresholds for any BGR color, useful for varying lighting conditions.
- Hue Wrapping for Red:
Handles HSV hue wrapping (0–10 and 170–180) to avoid missing red objects.
- Noise Reduction:
Combines GaussianBlur and morphological operations (MORPH_CLOSE, MORPH_OPEN) for clean masks.
- Real-Time Optimization:
Ignores small noise with contour area thresholding (area > 500).
- Mirror-Like View:
Flips frames (cv2.flip) for intuitive user interaction.
- Dynamic Highlighting:
Draws bounding boxes around detected objects; replace with contours for precise outlines.
- Fail-Safe Video Capture:
Handles webcam failures gracefully with if not ret.
- Multi-Color Support:
Extendable to detect multiple colors by blending masks.

---

# Day 3: Face Anonymizer  

This project implements face detection and blurring using **OpenCV** and **Mediapipe**. The application supports three different modes for processing images and video.
[Face Anonymizer output](Images/face-blur.png)

### Key Features:
- **Face Detection Model**: Uses Mediapipe's face detection model with a **50% confidence threshold** to detect faces.
- **Blurring**: A Gaussian blur with a kernel size of **(30, 30)** is applied to the detected faces.
- **Video Processing**: Processes video frames at **25 FPS** and saves them in an output directory.
- **Output**: All processed images and videos are saved in the **output** directory.
  
---

# Day 4: Text detector using tesseract

## **Features**
1. **Text Detection in Images**
   - Processes an input image to detect and extract text.
   - Highlights detected text regions with bounding boxes.
   
2. **Real-Time Text Detection via Webcam**
   - Captures webcam feed to detect and extract text in real-time.
   - Displays processed frames with detected text.


## **Key Concepts**
1. **Preprocessing Techniques**
   - Convert images to grayscale.
   - Apply thresholding to enhance text regions.
   - Use morphological operations (dilation) to emphasize text.

2. **Text Detection**
   - Use Tesseract OCR with custom configurations:
     - **OEM 3**: Tesseract engine mode for both legacy and LSTM models.
     - **PSM 6**: Page segmentation mode for detecting text blocks.
   - Extract text data (including bounding box coordinates) using `pytesseract.image_to_data`.

3. **Bounding Boxes**
   - Draw rectangles around detected text regions with confidence > 50%.
   
---

# Day 5: Parking Spot Detection and Classification

This project leverages Python and OpenCV to detect and classify parking spots in real-time. It processes video feeds or images to identify available and occupied parking spots,employing advanced image processing techniques to provide efficient parking management solutions. [Project](https://github.com/Ni2thin/30DaysOfComputerVision/tree/main/Projects/image-classification)

## **Features**
1. **Parking Spot Detection**  
   - Detect parking spots in video using a binary mask and extract regions of interest (ROIs).

2. **Real-Time Spot Classification**  
   - Classify each spot as `empty` or `not_empty` using a pre-trained SVM model.  
   - Update status dynamically and visualize with:
     - **Green** bounding boxes for `empty` spots.  
     - **Red** bounding boxes for `not_empty` spots.

3. **Availability Counter**  
   - Displays the number of available spots in real-time on the video feed.

4. **Efficient Frame Processing**  
   - Analyze frames at intervals to optimize performance while ensuring accuracy.

### **Output**
<img src="https://github.com/Ni2thin/30DaysOfComputerVision/blob/04c4f59c6f005878688dd18dac90a65fe234c853/Projects/image-classification/Parking%20spot%20detection/parking%20spot%20-%20output.png" width="1050" height="500"/>

### **Assets** 
[Download Dataset](https://drive.google.com/drive/folders/15lLq-6Bbuq7LyILMg2rzJOL4WpnxX6oX?usp=share_link) 

---
# Day 6: Emotion Detection with Python and OpenCV  

This project utilizes Python, OpenCV, and face landmark detection to classify human emotions from images. It processes face images, extracts 1404 key landmarks, and organizes them into labeled datasets for emotion classification. [Emotion detection project](https://github.com/Ni2thin/30DaysOfComputerVision/tree/main/Projects/Emotion%20detection)  

## **Features**  

- **Emotion Categorization**:  
  Processes images into seven predefined emotional categories (e.g., happy, sad, angry).  

- **Facial Landmark Extraction**:  
  Uses a custom `get_face_landmarks` function to capture precise facial landmarks essential for emotion detection.  

- **Batch Processing**:  
  Efficiently processes images in batches to optimize resource usage and avoid system overloads.  

- **Intermediate Data Storage**:  
  Saves labeled datasets for each emotion in separate text files (`data_<emotion>.txt`) and combines them into a single dataset (`data.txt`).  

- **Robust File Handling**:  
  Skips invalid or non-image files to ensure smooth processing.  

- **Scalable Architecture**:  
  Easily extendable to support more emotions or larger datasets without requiring significant code changes.  

- **Threading Optimization**:  
  Limits system thread usage to prevent crashes and improve stability, especially on macOS.  

---
# Day 7: Image Classification and Feature Extraction with Python  

This project leverages `Img2Vec` and a pre-trained Random Forest model to classify images into predefined categories. It extracts feature vectors from images, predicts their labels, and provides confidence scores for each classification.[Image classification and feature extraction project](https://github.com/Ni2thin/30DaysOfComputerVision/tree/main/Projects/img%20classification%20and%20extraction)

## **Features**  

- **Image Feature Extraction**:  
  Utilizes `Img2Vec` to generate 512-dimensional feature vectors, representing high-level characteristics of images.  

- **Robust Classification**:  
  Employs a Random Forest classifier to predict image labels with high accuracy.  

- **Confidence Scoring**:  
  Outputs confidence scores alongside predictions to gauge the reliability of the results.  

- **Error Handling**:  
  Includes checks for file existence and compatibility to prevent crashes during runtime.  

- **Human-Readable Output**:  
  Converts predictions into category labels (e.g., "Sunny", "Cloudy") for better interpretability.  

- **Flexible Input**:  
  Compatible with various image formats (e.g., JPG, PNG, BMP).  

- **Modular Workflow**:  
  Easily extendable to add more categories or integrate advanced models for improved classification.  

---
# Day 8: Building a Sign Language Detector with Python and OpenCV

This project implements a real-time sign language detection system using OpenCV, MediaPipe, and a machine learning model. The system captures hand gestures via a webcam, processes them to extract features, and predicts corresponding sign language characters. 
[Sign Language Detection Project](https://github.com/Ni2thin/30DaysOfComputerVision/tree/main/Projects/Sign%20language%20detection)


## Features

- **Dataset Creation**: Captures 100 images per class using the webcam for generating a labeled dataset.
- **Feature Extraction**: Utilizes MediaPipe's hand landmarks to extract normalized hand features (x, y coordinates).
- **Model Training**: Trains a Random Forest Classifier on the extracted features to distinguish between gestures.
- **Real-Time Prediction**: Detects hand gestures in live webcam feed and overlays the predicted character on the video stream.
- **Visualization**: Draws hand landmarks and bounding boxes around detected hands for intuitive visualization.
- **Expandable Design**: Easy to extend with additional gestures or classes by adding new labeled data.

## Workflow

### 1. Data Collection
- Webcam feed captures images for each gesture class.
- Images are saved to class-specific directories.

### 2. Model Training
- Extracts hand landmarks and normalizes them.
- Trains a Random Forest model on the features for classification.

### 3. Real-Time Detection
- Processes live webcam feed to detect and classify gestures.
- Displays predictions with bounding boxes and labels.

## Key Innovations

- **Hand Landmarks**: MediaPipe's hand tracking ensures accurate gesture representation.
- **Efficient Dataset Handling**: Automates data collection and preprocessing for rapid model training.
- **Dynamic Prediction**: Real-time responsiveness allows seamless interaction with the system.
- **Scalable Architecture**: Can be extended to support more complex sign language alphabets.

## How to Use

1. **Dataset Creation**:
   - Run the data collection script to capture hand gesture images.
   - Ensure each gesture has its own labeled folder in the dataset directory.

2. **Model Training**:
   - Run the training script to train the Random Forest Classifier.
   - A trained model will be saved for later use.

3. **Real-Time Detection**:
   - Launch the detection script to start the webcam feed.
   - The system will predict and display the corresponding gesture in real-time.

## Example Output

The system detects gestures and overlays predictions directly on the live video feed:
<p align="center">
  <img src="Projects/Sign%20language%20detection/outputs/B.png" alt="Image 1" height="300px">
  <img src="Projects/Sign%20language%20detection/outputs/A.png" alt="Image 2" height="300px">
</p>

---
# Day 9: Pneumonia Classification - Simple web app

This project utilizes Python and Keras to classify chest X-ray images as either showing signs of pneumonia or being normal. The model is trained using the Teachable Machine platform, and the dataset is taken from Kaggle's Chest X-Ray dataset. The goal is to assist in the identification of pneumonia cases using deep learning. [Project](https://github.com/Ni2thin/30DaysOfComputerVision/tree/main/Projects/Pneumonia%20classification)

## **Features**
1. **Pneumonia Detection**  
   - Classify chest X-ray images into two categories: `Pneumonia` and `Normal`.
   - Use pre-trained models fine-tuned with a custom dataset to provide accurate results.

2. **Prediction Confidence**  
   - Displays the model's confidence in the prediction as a percentage.
   - Provide a progress bar for an interactive user experience.

3. **Real-Time Upload and Prediction**  
   - Upload a chest X-ray image and get an immediate prediction for pneumonia classification.
   - Show prediction results on the app with a confidence score.

4. **User-Friendly Interface**  
   - The web app is designed with a simple and intuitive UI, allowing users to easily upload and classify chest X-ray images.
   - Streamlit is used to build an interactive and seamless user experience.

### **Output**
<p align="center">
  <img src="Projects/Pneumonia classification/output/output.png" alt="Image 1" height="300px">
  <img src="Projects/Pneumonia classification/output/output2.png" alt="Image 2" height="300px">
</p>

### **Assets**
- [Download Dataset from Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia/data)
- [Teachable Machine](https://teachablemachine.withgoogle.com/)
---
# Day 10: Object detection using AWS Rekognition 

This project demonstrates the use of AWS Rekognition to detect objects in a video stream. Specifically, it identifies instances of a target class (`Zebra` in this example) and saves the bounding box annotations and corresponding frames. The implementation leverages AWS Rekognition's label detection capabilities to process video frames and extract object information.

## **Features**
1. **Frame Processing and Object Detection**
   - Reads video frames using OpenCV.
   - Sends each frame to AWS Rekognition for object detection.
   - Identifies and processes instances of the target class (`Zebra`).

2. **Bounding Box Annotation**
   - Extracts bounding box details (center coordinates, width, and height) for the detected objects.
   - Saves annotations in YOLO-compatible format as text files for each frame.

3. **Frame Output Storage**
   - Saves each processed video frame as an image in a designated directory.
   - Frames are named sequentially for easy reference.

4. **Customizable Target Class**
   - Allows users to specify a different target object class by modifying the `target_class` variable.

5. **AWS Rekognition Integration**
   - Utilizes AWS Rekognition’s `detect_labels` API for high-confidence object detection.
   - Customizable minimum confidence threshold for detections.

## **Usage**
### Prerequisites
1. AWS Rekognition credentials:
   - Ensure you have an active AWS account with access to Rekognition services.
   - Replace the placeholders in `credentials.py` with your AWS Access Key and Secret Key:
     ```python
     access_key = "YOUR_AWS_ACCESS_KEY"
     secret_key = "YOUR_AWS_SECRET_KEY"
     ```

2. Install the required Python libraries:
   ```bash
   pip install boto3 opencv-python
   ```
   ---
# Day 11:  
   
   















