import cv2
import mediapipe as mp
import argparse
import os

# Function to process the image and detect faces
def process_img(img, face_detection):
    # Get the height (H) and width (W) of the image
    H, W, _ = img.shape

    # Convert the image from BGR (OpenCV default) to RGB (required by Mediapipe)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Use Mediapipe's face detection model to process the image
    out = face_detection.process(img_rgb)

    # If faces are detected, proceed with face blurring
    if out.detections is not None:
        for detection in out.detections:
            location_data = detection.location_data
            bbox = location_data.relative_bounding_box

            # Extract bounding box coordinates and dimensions (normalized)
            x1, y1, w, h = bbox.xmin, bbox.ymin, bbox.width, bbox.height

            # Convert normalized coordinates to pixel values based on image dimensions
            x1 = int(x1 * W)
            y1 = int(y1 * H)
            w = int(w * W)
            h = int(h * H)

            # Blur the face area in the image
            img[y1:y1 + h, x1:x1 + w, :] = cv2.blur(img[y1:y1 + h, x1:x1 + w, :], (30, 30))

    return img

# Argument parsing for selecting mode (image, video, webcam)
args = argparse.ArgumentParser()
args.add_argument("--mode", default='webcam')  # Can be 'image', 'video', or 'webcam'
args.add_argument("--filePath", default=None)  # File path for video/image (if mode is not webcam)
args = args.parse_args()

# Create output directory if it doesn't exist
output_dir = './output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Initialize Mediapipe's face detection model
mp_face_detection = mp.solutions.face_detection

# Start face detection model and process based on mode
with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:

    if args.mode == "image":
        # Read the image from the file path
        img = cv2.imread(args.filePath)
        # Process the image to blur detected faces
        img = process_img(img, face_detection)
        # Save the processed image to the output folder
        cv2.imwrite(os.path.join(output_dir, 'output.png'), img)

    elif args.mode == 'video':
        # Open the video file
        cap = cv2.VideoCapture(args.filePath)
        ret, frame = cap.read()

        # Initialize the video writer to save the output video
        output_video = cv2.VideoWriter(os.path.join(output_dir, 'output.mp4'),
                                       cv2.VideoWriter_fourcc(*'MP4V'),  # Codec for MP4 format
                                       25,  # Frame rate (25 FPS)
                                       (frame.shape[1], frame.shape[0]))  # Frame dimensions (width, height)

        # Process video frames until the end
        while ret:
            # Process the current frame to blur faces
            frame = process_img(frame, face_detection)
            # Write the processed frame to the output video
            output_video.write(frame)

            # Read the next frame
            ret, frame = cap.read()

        # Release video capture and writer objects after processing
        cap.release()
        output_video.release()

    elif args.mode == 'webcam':
        # Open the webcam feed
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()

        # Capture and process video frames in real-time
        while ret:
            # Flip the frame horizontally to create a mirror effect (optional)
            flipped_img = cv2.flip(frame, 1)
            # Process the flipped frame to blur faces
            frame = process_img(flipped_img, face_detection)

            # Display the processed frame
            cv2.imshow('frame', frame)

            # Wait for 25ms for a key press; 0xFF and ord('q') check if 'q' is pressed to quit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

            # Read the next frame
            ret, frame = cap.read()

        # Release the webcam feed and close the display window
        cap.release()
        cv2.destroyAllWindows()
