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
| Day 1 | 27-12-2024  | OpenCV tutorial for beginners                        | [YouTube: OpenCV tutorial for beginners](https://www.youtube.com/watch?v=eDIj5LuIL4A&list=PLb49csYFtO2HAdNGChGzohFJGnJnXBOqd&index=2)     |
| Day 2 | 28-12-2024  | Detecting color with Python and OpenCV               | [YouTube: Detecting color](link)                   |
| Day 3 | 29-12-2024  | Face detection and blurring with Python              | [YouTube: Face detection and blurring](link)       |
| Day 4 | 30-12-2024  | Text detection with Python | Tesseract             | [YouTube: Text detection with Python](link)        |
| Day 5 | 31-12-2024  | Image classification with Python and OpenCV          | [YouTube: Image classification with Python](link)  |
| Day 6 | 01-01-2025  | Emotion detection with Python, OpenCV, and others    | [YouTube: Emotion detection](link)                 |
| Day 7 | 02-01-2025  | Image classification + feature extraction           | [YouTube: Image classification + feature extraction](link) |
| Day 8 | 03-01-2025  | Sign language detection with Python and OpenCV       | [YouTube: Sign language detection](link)           |
| Day 9 | 04-01-2025  | Image classification WEB APP with Python & Flask     | [YouTube: Image classification web app](link)      |
| Day 10 | 05-01-2025 | AWS Rekognition tutorial | Object detection      | [YouTube: AWS Rekognition](link)                   |
| Day 11 | 06-01-2025 | Yolov8 object tracking 100% native                   | [YouTube: Yolov8 object tracking](link)            |
| Day 12 | 07-01-2025 | Image segmentation with Yolov8 custom dataset       | [YouTube: Image segmentation with Yolov8](link)    |
| Day 13 | 08-01-2025 | Train pose detection Yolov8 on custom dataset        | [YouTube: Train pose detection Yolov8](link)       |
| Day 14 | 09-01-2025 | Parking spot detection and counter                   | [YouTube: Parking spot detection](link)            |
| Day 15 | 10-01-2025 | Train Yolov10 object detection custom dataset        | [YouTube: Train Yolov10 object detection](link)    |
| Day 16 | 11-01-2025 | End to end pipeline real world computer vision       | [YouTube: End to end pipeline](link)               |
| Day 17 | 12-01-2025 | Image processing API with AWS API Gateway            | [YouTube: Image processing API](link)              |
| Day 18 | 13-01-2025 | How much data you need to train a computer vision model | [YouTube: How much data](link)                 |
| Day 19 | 14-01-2025 | Real world application of computer vision           | [YouTube: Real world application](link)            |
| Day 20 | 15-01-2025 | Train Detectron2 object detection custom dataset     | [YouTube: Train Detectron2](link)                  |
| Day 21 | 16-01-2025 | Face recognition on your webcam with Python & OpenCV | [YouTube: Face recognition](link)                  |
| Day 22 | 17-01-2025 | Face attendance + face recognition with OpenCV      | [YouTube: Face attendance](link)                   |
| Day 23 | 18-01-2025 | Machine learning with AWS practical projects        | [YouTube: ML with AWS](link)                       |
| Day 24 | 19-01-2025 | Chat with an image | LangChain custom                | [YouTube: Chat with an image](link)                |
| Day 25 | 20-01-2025 | Image generation with Python | Train Stable Diffusion | [YouTube: Image generation](link)                  |
| Day 26 | 21-01-2025 | Face recognition + liveness detection               | [YouTube: Face recognition + liveness](link)       |
| Day 27 | 22-01-2025 | Face recognition and face matching with Python      | [YouTube: Face matching](link)                     |
| Day 28 | 23-01-2025 | Machine learning web app with Python, Flask, and others | [YouTube: ML web app](link)                     |
| Day 29 | 24-01-2025 | Object detection on Raspberry Pi USB camera         | [YouTube: Object detection on Raspberry Pi](link)  |
| Day 30 | 25-01-2025 | Image generation with Python & Stable Diffusion     | [YouTube: Image generation](link)                  |


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

- Detect and analyze contours in images for shape analysis.
  
# Day 2: Detecting color with Python and OpenCV 

This project uses OpenCV to detect and track objects of a specific color in real-time using a webcam feed. It processes frames to isolate the target color, highlights detected objects with bounding boxes, and displays the result.

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




